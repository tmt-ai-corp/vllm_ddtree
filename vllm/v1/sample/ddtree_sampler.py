# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Target-side DDTree sampler."""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.sample.sampler import _SAMPLING_EPS, Sampler
from vllm.v1.spec_decode.ddtree import follow_verified_tree
from vllm.v1.spec_decode.metadata import DDTreeSpecDecodeMetadata


class DDTreeSampler(nn.Module):
    """Follow one target-accepted DDTree path per request.

    The fully general vLLM sampler path needs branch-aware grammar and repeated
    sampling metadata.  This class implements the exact greedy posterior path
    used by deterministic DDTree validation and records the accepted tree
    indices for KV and hidden-state compaction.
    """

    def __init__(self, sampler: Sampler, device: torch.device):
        super().__init__()
        self.sampler = sampler
        self.device = device

    def forward(
        self,
        metadata: DDTreeSpecDecodeMetadata,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        if sampling_metadata.temperature is not None:
            temperatures = sampling_metadata.temperature
            if torch.any(temperatures > _SAMPLING_EPS):
                raise NotImplementedError(
                    "DDTree currently supports greedy target posterior sampling "
                    "inside vLLM. Non-zero temperature requires repeated per-node "
                    "SamplingMetadata and branch grammar state."
                )
        if not sampling_metadata.no_penalties:
            raise NotImplementedError(
                "DDTree target posterior sampling with penalties needs "
                "branch-aware output-token histories."
            )
        if sampling_metadata.max_num_logprobs is not None:
            raise NotImplementedError(
                "DDTree logprobs require reordering per-node posterior logprobs "
                "into accepted-path output order."
            )
        if sampling_metadata.logprob_token_ids:
            raise NotImplementedError(
                "DDTree does not yet support per-token logprob requests."
            )
        if sampling_metadata.allowed_token_ids_mask is not None:
            raise NotImplementedError(
                "DDTree allowed-token masks need per-tree-node sampling metadata."
            )
        if sampling_metadata.bad_words_token_ids:
            raise NotImplementedError(
                "DDTree bad-words filtering needs per-tree-node sampling metadata."
            )
        if (
            sampling_metadata.logitsprocs.non_argmax_invariant
            or sampling_metadata.logitsprocs.argmax_invariant
        ):
            raise NotImplementedError(
                "DDTree logits processors need per-tree-node sampling metadata."
            )

        verify_logits = logits[metadata.target_logits_indices.long()].to(torch.float32)
        posterior = torch.argmax(verify_logits, dim=-1).to(torch.int32)

        batch_size = len(metadata.num_verify_tokens)
        max_output_len = metadata.max_verify_len
        output_token_ids = torch.full(
            (batch_size, max_output_len),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=logits.device,
        )
        accepted_indices_rows: list[list[int]] = []
        accepted_lengths: list[int] = []
        next_token_ids: list[int] = []
        bonus_logits_indices: list[int] = []

        start = 0
        for req_idx, verify_len in enumerate(metadata.num_verify_tokens):
            end = start + verify_len
            posterior_row = posterior[start:end]
            accepted_indices, next_token = follow_verified_tree(
                metadata.child_maps[req_idx], posterior_row
            )
            accepted_indices_rows.append(accepted_indices)
            accepted_lengths.append(len(accepted_indices))
            next_token_ids.append(next_token)

            # Visible outputs exclude the root and append the final bonus token.
            visible_tokens: list[int] = []
            for tree_index in accepted_indices[1:]:
                visible_tokens.append(
                    int(metadata.verify_input_ids[start + tree_index].item())
                )
            visible_tokens.append(next_token)
            if visible_tokens:
                output_token_ids[
                    req_idx, : len(visible_tokens)
                ] = torch.tensor(
                    visible_tokens, dtype=torch.int32, device=logits.device
                )

            final_tree_index = accepted_indices[-1]
            bonus_logits_indices.append(start + final_tree_index)
            start = end

        max_accepted = max(accepted_lengths) if accepted_lengths else 0
        accepted_indices_gpu = torch.full(
            (batch_size, max_accepted),
            -1,
            dtype=torch.int32,
            device=logits.device,
        )
        for req_idx, accepted in enumerate(accepted_indices_rows):
            accepted_indices_gpu[req_idx, : len(accepted)] = torch.tensor(
                accepted, dtype=torch.int32, device=logits.device
            )
        metadata.accepted_indices_gpu = accepted_indices_gpu
        metadata.accepted_lengths_gpu = torch.tensor(
            accepted_lengths, dtype=torch.int32, device=logits.device
        )
        metadata.next_token_ids_gpu = torch.tensor(
            next_token_ids, dtype=torch.int32, device=logits.device
        )
        metadata.bonus_logits_indices = torch.tensor(
            bonus_logits_indices, dtype=torch.int32, device=logits.device
        )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )
