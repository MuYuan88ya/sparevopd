from types import SimpleNamespace

import pytest
import torch

from verl.utils.teacher_visual_token_mask import create_visual_token_selection_strategy
from verl.workers.actor.dp_actor import DataParallelPPOActor


def make_actor(*, background_keep_ratio: float = 0.1) -> DataParallelPPOActor:
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=99,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
    )
    actor.teacher_visual_token_selection_strategy = create_visual_token_selection_strategy(
        "fixed_stratified",
        {"background_keep_ratio": background_keep_ratio, "seed": 0},
    )
    return actor


def make_inputs():
    # Merged visual grids are 2x2=4 and 2x3=6, respectively.
    multi_modal_inputs = [
        {"image_grid_thw": torch.tensor([[1, 4, 4]])},
        {"image_grid_thw": torch.tensor([[1, 4, 6]])},
    ]
    selection_metadata = [
        {"bbox": [0, 0, 2, 2], "image_width": 4, "image_height": 4, "sample_id": "first"},
        {"bbox": [0, 0, 3, 4], "image_width": 6, "image_height": 4, "sample_id": "second"},
    ]
    dense_image_embeds = torch.randn(10, 8)
    input_ids = torch.tensor(
        [
            [1, 99, 99, 99, 99, 7, 8, 0, 0, 0],
            [1, 99, 99, 99, 99, 99, 99, 7, 8, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    teacher_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_start_idx": torch.tensor([6, 8]),
        "multi_modal_inputs": multi_modal_inputs,
    }
    return multi_modal_inputs, selection_metadata, dense_image_embeds, teacher_inputs


def test_teacher_consumes_student_state_for_heterogeneous_batch_without_mutating_student_inputs():
    actor = make_actor()
    multi_modal_inputs, selection_metadata, dense_image_embeds, teacher_inputs = make_inputs()

    state = actor._build_visual_token_selection_state(
        dense_image_embeds,
        multi_modal_inputs,
        selection_metadata,
        batch_size=2,
    )
    finalized, metrics = actor._finalize_teacher_visual_token_mask(teacher_inputs, state)

    assert state["token_counts"] == [4, 6]
    assert [int(mask.sum()) for mask in state["keep_masks"]] == [2, 5]
    assert all("image_token_keep_mask" not in inputs for inputs in multi_modal_inputs)
    assert all(
        torch.equal(final_inputs["image_token_keep_mask"], state_mask)
        for final_inputs, state_mask in zip(finalized["multi_modal_inputs"], state["keep_masks"], strict=True)
    )
    assert finalized["attention_mask"][0].tolist() == [1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
    assert finalized["attention_mask"][1].tolist() == [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    assert torch.equal(finalized["attention_mask"][0, 6:], teacher_inputs["attention_mask"][0, 6:])
    assert torch.equal(finalized["attention_mask"][1, 8:], teacher_inputs["attention_mask"][1, 8:])
    assert metrics["actor/teacher_kept_visual_tokens"] == 7.0
    assert metrics["actor/teacher_llm_effective_tokens"] == float(finalized["attention_mask"].sum())


def test_teacher_rejects_state_grid_or_batch_mismatch():
    actor = make_actor(background_keep_ratio=0.0)
    # Build a valid state first, then corrupt only the authoritative token count.
    multi_modal_inputs, metadata, dense_image_embeds, teacher_inputs = make_inputs()
    state = actor._build_visual_token_selection_state(dense_image_embeds, multi_modal_inputs, metadata, 2)
    state["token_counts"][1] = 4
    with pytest.raises(ValueError, match="token count"):
        actor._finalize_teacher_visual_token_mask(teacher_inputs, state)

    state = actor._build_visual_token_selection_state(dense_image_embeds, multi_modal_inputs, metadata, 2)
    state["keep_masks"].pop()
    with pytest.raises(ValueError, match="one entry per teacher sample"):
        actor._finalize_teacher_visual_token_mask(teacher_inputs, state)
