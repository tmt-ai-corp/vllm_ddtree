# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV cache helpers for DDTree verification windows."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DDTreeKVCommitPlan:
    req_ids: list[str]
    src_slot_mapping_by_group: dict[int, torch.Tensor]
    dst_slot_mapping_by_group: dict[int, torch.Tensor]
    accepted_indices_by_req: list[list[int]]
    accepted_lengths: list[int]
    verify_lengths: list[int]
    num_rejected_tree_nodes: list[int]
    kv_cache_tensors: list[torch.Tensor]
    kv_cache_config: object | None = None


def _flatten_attention_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    """Return a view with token slot as dimension 1.

    vLLM attention caches commonly use ``[2, num_blocks, block, ...]`` or
    ``[num_blocks, 2, block, ...]``.  The returned tensor is
    ``[2, num_slots, ...]`` so raw slot ids can be used without interpreting
    logical positions.
    """

    if kv_cache.ndim < 4:
        raise NotImplementedError(
            "DDTree KV compaction currently supports attention KV tensors with "
            "layout [2, num_blocks, block_size, ...] or "
            "[num_blocks, 2, block_size, ...]."
        )
    if kv_cache.shape[0] == 2:
        return kv_cache.reshape(
            2, kv_cache.shape[1] * kv_cache.shape[2], *kv_cache.shape[3:]
        )
    if kv_cache.shape[1] == 2:
        cache = kv_cache.transpose(0, 1)
        return cache.reshape(2, cache.shape[1] * cache.shape[2], *cache.shape[3:])
    raise NotImplementedError(
        "DDTree KV compaction could not identify the K/V dimension in a cache "
        f"with shape {tuple(kv_cache.shape)}."
    )


def compact_kv_cache_by_slots(
    kv_cache_tensors: list[torch.Tensor],
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
) -> None:
    """Copy accepted KV slots to contiguous destination slots in-place."""

    if src_slots.numel() != dst_slots.numel():
        raise ValueError(
            f"src_slots and dst_slots must have the same length, got "
            f"{src_slots.numel()} and {dst_slots.numel()}."
        )
    if src_slots.numel() == 0:
        return

    seen_storage: set[tuple[int, int]] = set()
    for kv_cache in kv_cache_tensors:
        if not torch.is_tensor(kv_cache):
            continue
        storage_key = (kv_cache.untyped_storage().data_ptr(), kv_cache.storage_offset())
        if storage_key in seen_storage:
            continue
        seen_storage.add(storage_key)

        flat = _flatten_attention_kv_cache(kv_cache)
        src = src_slots.to(device=flat.device, dtype=torch.long, non_blocking=True)
        dst = dst_slots.to(device=flat.device, dtype=torch.long, non_blocking=True)
        kept = flat.index_select(1, src)
        flat.index_copy_(1, dst, kept)


def build_slot_compaction_for_request(
    verify_slot_mapping: torch.Tensor,
    accepted_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build source/destination slots for one flat verify window."""

    if not accepted_indices:
        return (
            torch.empty(0, dtype=torch.long, device=verify_slot_mapping.device),
            torch.empty(0, dtype=torch.long, device=verify_slot_mapping.device),
        )
    accepted = torch.tensor(
        accepted_indices, dtype=torch.long, device=verify_slot_mapping.device
    )
    src = verify_slot_mapping.index_select(0, accepted)
    dst = verify_slot_mapping[: len(accepted_indices)]
    return src, dst
