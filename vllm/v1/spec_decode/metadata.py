# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )


@dataclass
class DDTreeSpecDecodeMetadata:
    is_ddtree: Literal[True]
    num_verify_tokens: list[int]
    cu_num_verify_tokens: torch.Tensor
    # All verify token logits, in packed logits tensor coordinates.
    target_logits_indices: torch.Tensor
    logits_indices: torch.Tensor
    # Initialized to the last verify node per request. The DDTree sampler stores
    # the final accepted-node logits index after path following.
    bonus_logits_indices: torch.Tensor
    verify_input_ids: torch.Tensor
    verify_position_ids: torch.Tensor
    verify_slot_mapping: torch.Tensor | dict[str, torch.Tensor]
    tree_visibility: torch.Tensor
    tree_start_offsets: torch.Tensor
    tree_lengths: torch.Tensor
    child_maps: list[list[dict[int, int]]]
    parents: list[list[int]]
    node_token_ids_by_req: list[list[int]]
    node_depths_by_req: list[list[int]]
    accepted_indices_gpu: torch.Tensor | None = None
    accepted_lengths_gpu: torch.Tensor | None = None
    next_token_ids_gpu: torch.Tensor | None = None

    def __post_init__(self):
        self.max_spec_len = max(self.num_verify_tokens) - 1
        self.max_verify_len = max(self.num_verify_tokens)

    @classmethod
    def make_dummy(
        cls,
        device: torch.device,
        num_reqs: int = 1,
        verify_len: int = 1,
    ) -> "DDTreeSpecDecodeMetadata":
        num_verify_tokens = [verify_len] * num_reqs
        cu_num_verify = np.cumsum(num_verify_tokens, dtype=np.int32)
        total = int(cu_num_verify[-1])
        cu_tensor = torch.from_numpy(cu_num_verify).to(device)
        logits_indices = torch.arange(total, dtype=torch.int32, device=device)
        bonus = cu_tensor - 1
        visibility = torch.zeros(
            (num_reqs, verify_len, verify_len), dtype=torch.bool, device=device
        )
        visibility[:, 0, 0] = True
        return cls(
            is_ddtree=True,
            num_verify_tokens=num_verify_tokens,
            cu_num_verify_tokens=cu_tensor,
            target_logits_indices=logits_indices,
            logits_indices=logits_indices,
            bonus_logits_indices=bonus,
            verify_input_ids=torch.zeros(total, dtype=torch.int32, device=device),
            verify_position_ids=torch.zeros(total, dtype=torch.long, device=device),
            verify_slot_mapping=torch.zeros(total, dtype=torch.long, device=device),
            tree_visibility=visibility,
            tree_start_offsets=torch.arange(
                num_reqs, dtype=torch.int32, device=device
            )
            * verify_len,
            tree_lengths=torch.full(
                (num_reqs,), verify_len, dtype=torch.int32, device=device
            ),
            child_maps=[[dict()] for _ in range(num_reqs)],
            parents=[[-1] for _ in range(num_reqs)],
            node_token_ids_by_req=[[] for _ in range(num_reqs)],
            node_depths_by_req=[[] for _ in range(num_reqs)],
        )
