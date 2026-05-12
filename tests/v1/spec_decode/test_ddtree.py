# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

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
