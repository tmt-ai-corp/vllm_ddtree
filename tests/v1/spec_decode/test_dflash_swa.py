# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.transformers_utils.configs.speculators import SpeculatorsConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.spec_decode import dflash as dflash_module
from vllm.v1.spec_decode.dflash import DFlashProposer


class _FakeBuilder:
    def __init__(
        self, kv_cache_spec=None, layer_names=None, vllm_config=None, device=None
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names

    def build_for_drafting(self, common_attn_metadata, draft_index):
        return SimpleNamespace(causal=common_attn_metadata.causal)


class _FakeBackend:
    @classmethod
    def full_cls_name(cls):
        return "fake.backend"

    @classmethod
    def get_builder_cls(cls):
        return _FakeBuilder


class _FakeLayer:
    def get_attn_backend(self):
        return _FakeBackend


class _FakeAttentionGroup:
    def __init__(self, layer_names):
        self.layer_names = layer_names
        self._builder = _FakeBuilder()

    def get_metadata_builder(self):
        return self._builder


def test_dflash_speculators_preserves_swa_config():
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    config = {
        "speculators_model_type": "dflash",
        "transformer_layer_config": {
            "num_hidden_layers": len(layer_types),
            "sliding_window": None,
        },
        "draft_vocab_size": 100,
        "target_hidden_size": 64,
        "aux_hidden_state_layer_ids": [0, 1, 2],
        "mask_token_id": 99,
        "layer_types": layer_types,
        "use_sliding_window": True,
        "sliding_window": 2048,
        "max_window_layers": len(layer_types),
    }

    hf_config = SpeculatorsConfig.extract_transformers_pre_trained_config(config)

    assert hf_config["layer_types"] == layer_types
    assert hf_config["use_sliding_window"] is True
    assert hf_config["sliding_window"] == 2048
    assert hf_config["max_window_layers"] == len(layer_types)


def test_dflash_swa_layers_use_causal_metadata():
    proposer = object.__new__(DFlashProposer)
    proposer.model = SimpleNamespace(sliding_attention_layer_names={"layer.sw"})
    proposer.draft_attn_groups = [
        _FakeAttentionGroup(["layer.sw"]),
        _FakeAttentionGroup(["layer.full"]),
    ]
    cad = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=torch.empty(1, 1, dtype=torch.int32),
        slot_mapping=torch.empty(2, dtype=torch.int64),
        causal=False,
    )

    per_group, per_layer = DFlashProposer.build_per_group_and_layer_attn_metadata(
        proposer, cad
    )

    assert per_group[0].causal is True
    assert per_group[1].causal is False
    assert per_layer["layer.sw"].causal is True
    assert per_layer["layer.full"].causal is False


def test_dflash_initializes_split_swa_draft_groups(monkeypatch):
    proposer = object.__new__(DFlashProposer)
    proposer.vllm_config = SimpleNamespace()
    proposer.device = torch.device("cpu")
    proposer.model = SimpleNamespace(sliding_attention_layer_names={"layer.sw"})
    proposer._draft_attn_layer_names = {"layer.sw", "layer.full"}

    monkeypatch.setattr(
        dflash_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {
            "layer.sw": _FakeLayer(),
            "layer.full": _FakeLayer(),
        },
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer.sw"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                    sliding_window=4,
                ),
            ),
            KVCacheGroupSpec(
                ["layer.full"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                ),
            ),
        ],
    )

    DFlashProposer.initialize_attn_backend(proposer, kv_cache_config, [16, 16])

    assert len(proposer.draft_attn_groups) == 2
    assert {group.kv_cache_group_id for group in proposer.draft_attn_groups} == {0, 1}
    assert {tuple(group.layer_names) for group in proposer.draft_attn_groups} == {
        ("layer.sw",),
        ("layer.full",),
    }
