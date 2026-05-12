# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    make_kv_sharing_fast_prefill_common_attn_metadata,
)
from vllm.v1.spec_decode.ddtree import (
    build_ddtree_proposal,
    follow_verified_tree,
    pack_ddtree_proposals,
)
from vllm.v1.spec_decode.ddtree_kv import compact_kv_cache_by_slots


def test_ddtree_build_and_follow_path():
    logits = torch.full((3, 6), -10.0)
    logits[0, 1] = 4.0
    logits[0, 2] = 3.0
    logits[1, 3] = 5.0
    logits[1, 4] = 1.0
    logits[2, 5] = 6.0

    proposal = build_ddtree_proposal(logits, budget=4)

    assert proposal.num_nodes == 4
    assert proposal.parents[0] == -1
    assert proposal.visibility.shape == (5, 5)
    assert proposal.visibility.diag().all()

    posterior = torch.tensor([1, 3, 5, 0, 0], dtype=torch.int64)
    accepted_indices, next_token = follow_verified_tree(
        proposal.child_maps, posterior
    )
    assert accepted_indices == [0, 1, 2, 3]
    assert next_token == 0


def test_pack_ddtree_proposals_repeats_positions_but_not_slots():
    logits = torch.full((2, 8), -10.0)
    logits[0, 1] = 4.0
    logits[0, 2] = 3.0
    logits[1, 3] = 5.0
    proposal = build_ddtree_proposal(logits, budget=3)

    packed = pack_ddtree_proposals(
        [11], [7], [proposal], device=torch.device("cpu")
    )

    assert packed.flat_verify_input_ids[0].item() == 11
    assert packed.flat_verify_position_ids[0].item() == 7
    assert packed.flat_verify_position_ids.tolist()[1:] == [
        7 + depth for depth in proposal.node_depths.tolist()
    ]
    assert packed.flat_verify_tree_indices.tolist() == [0, 1, 2, 3]


def test_compact_kv_cache_by_slots_keeps_accepted_slots():
    kv_cache = torch.arange(2 * 2 * 4 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 4, 1, 1
    )
    before = kv_cache.clone()

    compact_kv_cache_by_slots(
        [kv_cache],
        src_slots=torch.tensor([4, 7], dtype=torch.long),
        dst_slots=torch.tensor([4, 5], dtype=torch.long),
    )

    flat_before = before.reshape(2, 8, 1, 1)
    flat_after = kv_cache.reshape(2, 8, 1, 1)
    torch.testing.assert_close(flat_after[:, 4], flat_before[:, 4])
    torch.testing.assert_close(flat_after[:, 5], flat_before[:, 7])


def test_kv_sharing_fast_prefill_preserves_ddtree_metadata():
    positions = torch.tensor([10, 11, 11, 20, 21], dtype=torch.long)
    visibility = torch.zeros((2, 3, 3), dtype=torch.bool)
    visibility[0, :3, :3] = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, False, True],
        ]
    )
    visibility[1, :2, :2] = torch.tensor(
        [
            [True, False],
            [True, True],
        ]
    )
    tree_lengths = torch.tensor([3, 2], dtype=torch.int32)
    logits_indices = torch.arange(5, dtype=torch.int32)

    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 5], dtype=torch.int32),
        seq_lens=torch.tensor([13, 22], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=22,
        block_table_tensor=torch.zeros((2, 2), dtype=torch.int32),
        slot_mapping=torch.arange(5, dtype=torch.long),
        logits_indices_padded=logits_indices,
        num_logits_indices=5,
        positions=positions,
        ddtree_visibility=visibility,
        ddtree_tree_lengths=tree_lengths,
        ddtree_position_ids=positions,
    )

    repacked = make_kv_sharing_fast_prefill_common_attn_metadata(metadata)

    assert repacked.ddtree_visibility is visibility
    assert repacked.ddtree_tree_lengths is tree_lengths
    torch.testing.assert_close(repacked.ddtree_position_ids, positions)
    torch.testing.assert_close(repacked.positions, positions)
