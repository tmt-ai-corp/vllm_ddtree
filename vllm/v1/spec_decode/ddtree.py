# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DDTree helpers for DFlash speculative decoding.

This module intentionally mirrors the heap-based reference implementation in
``/home/tmtaicorp/LittleD/ddtree/ddtree.py``.  The tree is built from
target-vocab draft logits, flattened in heap pop order, and verified by
following target posterior tokens through per-node child maps.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


DDTREE_TREE_BUILD_STAGE_ORDER = (
    "tree_build_copy",
    "tree_build_heap",
    "tree_build_visibility",
)


def _empty_tree_build_times() -> dict[str, float]:
    return {name: 0.0 for name in DDTREE_TREE_BUILD_STAGE_ORDER}


@dataclass
class DDTreeRequestProposal:
    """A single request's DDTree proposal.

    ``node_token_ids`` and ``node_depths`` exclude the root.  ``parents`` and
    ``child_maps`` include the root at index 0, so tree node indices are:
    root=0, first non-root node=1.
    """

    node_token_ids: list[int]
    node_depths: list[int]
    parents: list[int]
    child_maps: list[dict[int, int]]
    visibility: list[list[bool]]
    build_times: dict[str, float] | None = None

    @property
    def num_nodes(self) -> int:
        return len(self.node_token_ids)

    @property
    def verify_length(self) -> int:
        return 1 + self.num_nodes

    def to_tensors(
        self, device: torch.device | None = None
    ) -> "DDTreeRequestProposalTensors":
        return DDTreeRequestProposalTensors(
            node_token_ids=torch.tensor(
                self.node_token_ids, dtype=torch.long, device=device
            ),
            node_depths=torch.tensor(self.node_depths, dtype=torch.long, device=device),
            parents=self.parents,
            child_maps=self.child_maps,
            visibility=torch.tensor(self.visibility, dtype=torch.bool, device=device),
            build_times=self.build_times,
        )

    @classmethod
    def from_tensors(
        cls,
        node_token_ids: torch.Tensor,
        node_depths: torch.Tensor,
        parents: list[int],
        child_maps: list[dict[int, int]],
        visibility: torch.Tensor,
        build_times: dict[str, float] | None = None,
    ) -> "DDTreeRequestProposal":
        return cls(
            node_token_ids=[int(x) for x in node_token_ids.detach().cpu().tolist()],
            node_depths=[int(x) for x in node_depths.detach().cpu().tolist()],
            parents=[int(x) for x in parents],
            child_maps=[
                {int(token): int(index) for token, index in child_map.items()}
                for child_map in child_maps
            ],
            visibility=[
                [bool(v) for v in row]
                for row in visibility.detach().cpu().tolist()
            ],
            build_times=build_times,
        )


@dataclass
class DDTreeRequestProposalTensors:
    node_token_ids: torch.Tensor
    node_depths: torch.Tensor
    parents: list[int]
    child_maps: list[dict[int, int]]
    visibility: torch.Tensor
    build_times: dict[str, float] | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.node_token_ids.numel())

    @property
    def verify_length(self) -> int:
        return 1 + self.num_nodes

    def to_cpu_proposal(self) -> DDTreeRequestProposal:
        return DDTreeRequestProposal.from_tensors(
            self.node_token_ids,
            self.node_depths,
            self.parents,
            self.child_maps,
            self.visibility,
            self.build_times,
        )


@dataclass
class DDTreeProposalBatch:
    """Batch of DDTree proposals produced by the DFlash proposer."""

    proposals: list[DDTreeRequestProposalTensors]

    def to_cpu(self) -> list[DDTreeRequestProposal]:
        return [proposal.to_cpu_proposal() for proposal in self.proposals]

    def __len__(self) -> int:
        return len(self.proposals)


@dataclass
class PackedDDTreeBatch:
    flat_verify_input_ids: torch.Tensor
    flat_verify_position_ids: torch.Tensor
    flat_verify_tree_indices: torch.Tensor
    flat_verify_request_indices: torch.Tensor
    cu_verify_tokens: torch.Tensor
    num_verify_tokens: list[int]
    max_verify_tokens: int
    visibility_blocks: torch.Tensor
    node_depths_by_req: list[list[int]]
    child_maps_by_req: list[list[dict[int, int]]]
    parents_by_req: list[list[int]]
    node_token_ids_by_req: list[list[int]]
    tree_start_offsets: torch.Tensor
    tree_lengths: torch.Tensor


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
    dict[str, float],
]:
    """Build a probability-weighted DDTree from draft logits.

    Args:
        draft_logits: ``[draft_horizon, vocab]`` target-vocab logits.
        budget: Number of non-root tree nodes to keep.

    Returns:
        ``node_token_ids``, ``node_depths``, ``parents``, ``child_maps``,
        ``visibility``, and build sub-stage timings.
    """

    build_subtimes = _empty_tree_build_times()
    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
        )

    topk = min(int(budget), int(draft_logits.shape[-1]))
    depth_limit = int(draft_logits.shape[0])

    copy_start = time.perf_counter()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    build_subtimes["tree_build_copy"] = time.perf_counter() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    heap_start = time.perf_counter()
    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (
                    -sibling_logw,
                    sibling_ranks,
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw),
            )

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility, build_subtimes


def build_ddtree_proposal(
    draft_logits: torch.Tensor,
    budget: int,
) -> DDTreeRequestProposalTensors:
    (
        node_token_ids,
        node_depths,
        parents,
        child_maps,
        visibility,
        build_times,
    ) = build_ddtree_tree(draft_logits, budget)
    return DDTreeRequestProposalTensors(
        node_token_ids=node_token_ids,
        node_depths=node_depths,
        parents=parents,
        child_maps=child_maps,
        visibility=visibility,
        build_times=build_times,
    )


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior: torch.Tensor,
) -> tuple[list[int], int]:
    """Follow the accepted root-to-leaf path from target posterior tokens."""

    posterior_tokens = posterior.reshape(-1).tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def pack_ddtree_proposals(
    root_token_ids: list[int] | torch.Tensor,
    start_positions: list[int] | torch.Tensor,
    proposals: list[DDTreeRequestProposal | DDTreeRequestProposalTensors],
    *,
    device: torch.device,
) -> PackedDDTreeBatch:
    """Pack request proposals into flat vLLM-shaped verify tensors."""

    if torch.is_tensor(root_token_ids):
        roots = [int(x) for x in root_token_ids.detach().cpu().tolist()]
    else:
        roots = [int(x) for x in root_token_ids]
    if torch.is_tensor(start_positions):
        starts = [int(x) for x in start_positions.detach().cpu().tolist()]
    else:
        starts = [int(x) for x in start_positions]

    if len(roots) != len(proposals) or len(starts) != len(proposals):
        raise ValueError(
            "root_token_ids, start_positions, and proposals must have the same length"
        )

    input_ids: list[int] = []
    position_ids: list[int] = []
    tree_indices: list[int] = []
    request_indices: list[int] = []
    cu_verify: list[int] = []
    visibility_tensors: list[torch.Tensor] = []
    node_depths_by_req: list[list[int]] = []
    child_maps_by_req: list[list[dict[int, int]]] = []
    parents_by_req: list[list[int]] = []
    node_token_ids_by_req: list[list[int]] = []

    running = 0
    max_verify_tokens = 0
    for req_idx, (root, start, proposal) in enumerate(zip(roots, starts, proposals)):
        if isinstance(proposal, DDTreeRequestProposal):
            proposal_t = proposal.to_tensors()
        else:
            proposal_t = proposal

        node_token_ids = [
            int(x) for x in proposal_t.node_token_ids.detach().cpu().tolist()
        ]
        node_depths = [int(x) for x in proposal_t.node_depths.detach().cpu().tolist()]
        verify_len = 1 + len(node_token_ids)
        max_verify_tokens = max(max_verify_tokens, verify_len)

        input_ids.append(root)
        input_ids.extend(node_token_ids)
        position_ids.append(start)
        position_ids.extend(start + depth for depth in node_depths)
        tree_indices.extend(range(verify_len))
        request_indices.extend([req_idx] * verify_len)
        running += verify_len
        cu_verify.append(running)
        visibility_tensors.append(proposal_t.visibility.to(dtype=torch.bool))
        node_depths_by_req.append(node_depths)
        child_maps_by_req.append(proposal_t.child_maps)
        parents_by_req.append(proposal_t.parents)
        node_token_ids_by_req.append(node_token_ids)

    batch_size = len(proposals)
    visibility_blocks = torch.zeros(
        (batch_size, max_verify_tokens, max_verify_tokens),
        dtype=torch.bool,
        device=device,
    )
    for req_idx, visibility in enumerate(visibility_tensors):
        length = visibility.shape[0]
        visibility_blocks[req_idx, :length, :length].copy_(
            visibility.to(device=device, non_blocking=True)
        )

    return PackedDDTreeBatch(
        flat_verify_input_ids=torch.tensor(input_ids, dtype=torch.int32, device=device),
        flat_verify_position_ids=torch.tensor(
            position_ids, dtype=torch.long, device=device
        ),
        flat_verify_tree_indices=torch.tensor(
            tree_indices, dtype=torch.int32, device=device
        ),
        flat_verify_request_indices=torch.tensor(
            request_indices, dtype=torch.int32, device=device
        ),
        cu_verify_tokens=torch.tensor(cu_verify, dtype=torch.int32, device=device),
        num_verify_tokens=[
            (cu_verify[i] - (cu_verify[i - 1] if i > 0 else 0))
            for i in range(len(cu_verify))
        ],
        max_verify_tokens=max_verify_tokens,
        visibility_blocks=visibility_blocks,
        node_depths_by_req=node_depths_by_req,
        child_maps_by_req=child_maps_by_req,
        parents_by_req=parents_by_req,
        node_token_ids_by_req=node_token_ids_by_req,
        tree_start_offsets=torch.tensor(
            [0] + cu_verify[:-1], dtype=torch.int32, device=device
        ),
        tree_lengths=torch.tensor(
            [
                (cu_verify[i] - (cu_verify[i - 1] if i > 0 else 0))
                for i in range(len(cu_verify))
            ],
            dtype=torch.int32,
            device=device,
        ),
    )


def clone_proposal_for_msgspec(proposal: DDTreeRequestProposal) -> dict[str, Any]:
    """Return a plain-container copy suitable for msgpack/msgspec transport."""

    return {
        "node_token_ids": list(proposal.node_token_ids),
        "node_depths": list(proposal.node_depths),
        "parents": list(proposal.parents),
        "child_maps": [
            {int(token): int(index) for token, index in child_map.items()}
            for child_map in proposal.child_maps
        ],
        "visibility": [[bool(v) for v in row] for row in proposal.visibility],
        "build_times": dict(proposal.build_times or {}),
    }


def proposal_from_plain(
    data: DDTreeRequestProposal | dict[str, Any],
) -> DDTreeRequestProposal:
    if isinstance(data, DDTreeRequestProposal):
        return data
    return DDTreeRequestProposal(
        node_token_ids=[int(x) for x in data["node_token_ids"]],
        node_depths=[int(x) for x in data["node_depths"]],
        parents=[int(x) for x in data["parents"]],
        child_maps=[
            {int(token): int(index) for token, index in child_map.items()}
            for child_map in data["child_maps"]
        ],
        visibility=[[bool(v) for v in row] for row in data["visibility"]],
        build_times={
            str(key): float(value)
            for key, value in (data.get("build_times") or {}).items()
        },
    )
