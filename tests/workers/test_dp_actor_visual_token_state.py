from types import SimpleNamespace

import pytest
import torch

from verl.utils.teacher_visual_token_mask import create_visual_token_selection_strategy
from verl.workers.actor.dp_actor import DataParallelPPOActor


def make_tiny_qwen3_5():
    try:
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    except ImportError:
        pytest.skip("Qwen3.5 is unavailable in this transformers build")

    text_config = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "layer_types": ["full_attention"],
        "max_position_embeddings": 64,
        "use_cache": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "pad_token_id": 0,
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 4,
        "out_hidden_size": 32,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 2304,
    }
    return Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=99,
            video_token_id=100,
        )
    )


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
    return multi_modal_inputs, selection_metadata, teacher_inputs


def test_teacher_consumes_student_state_for_heterogeneous_batch_without_mutating_student_inputs():
    actor = make_actor()
    multi_modal_inputs, selection_metadata, teacher_inputs = make_inputs()

    _, image_grid_thw, target_keep_masks, sample_ids = actor._prepare_visual_token_selection_inputs(
        multi_modal_inputs,
        selection_metadata,
        batch_size=2,
    )
    state = actor._build_visual_token_selection_state_from_keep_mask(
        torch.tensor(
            [True, False, True, False, True, True, False, True, True, True],
            dtype=torch.bool,
        ),
        image_grid_thw,
        target_keep_masks,
        sample_ids,
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
    multi_modal_inputs, metadata, teacher_inputs = make_inputs()
    _, image_grid_thw, target_keep_masks, sample_ids = actor._prepare_visual_token_selection_inputs(
        multi_modal_inputs,
        metadata,
        batch_size=2,
    )
    state = actor._build_visual_token_selection_state_from_keep_mask(
        torch.ones(10, dtype=torch.bool), image_grid_thw, target_keep_masks, sample_ids
    )
    state["token_counts"][1] = 4
    with pytest.raises(ValueError, match="token count"):
        actor._finalize_teacher_visual_token_mask(teacher_inputs, state)

    state = actor._build_visual_token_selection_state_from_keep_mask(
        torch.ones(10, dtype=torch.bool), image_grid_thw, target_keep_masks, sample_ids
    )
    state["keep_masks"].pop()
    with pytest.raises(ValueError, match="one entry per teacher sample"):
        actor._finalize_teacher_visual_token_mask(teacher_inputs, state)


def test_full_student_to_teacher_flow_consumes_one_mask_per_image(monkeypatch):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5Model

    import verl.utils.npu_flash_attn_utils as npu_flash_attn_utils
    import verl.workers.actor.dp_actor as dp_actor_module
    from verl.models.transformers.qwen3_5 import forward_with_normal_backend, qwen3_5_base_forward

    monkeypatch.setattr(Qwen3_5Model, "forward", qwen3_5_base_forward)
    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "forward", forward_with_normal_backend)
    monkeypatch.setattr(dp_actor_module, "unpad_input", npu_flash_attn_utils.unpad_input)
    monkeypatch.setattr(dp_actor_module, "pad_input", npu_flash_attn_utils.pad_input)
    monkeypatch.setattr(dp_actor_module, "index_first_axis", npu_flash_attn_utils.index_first_axis)
    monkeypatch.setattr(dp_actor_module, "rearrange", npu_flash_attn_utils.rearrange)

    model = make_tiny_qwen3_5().eval()
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = {
        "calculate_sum_pi_squared": False,
        "sum_pi_squared_checkpointing": False,
        "entropy_checkpointing": False,
    }
    actor.actor_module = model
    actor.teacher_visual_token_selection_strategy = create_visual_token_selection_strategy(
        "fixed_stratified", {"background_keep_ratio": 0.1, "seed": 0}
    )
    actor.use_prefix_grouper = False
    actor.use_remove_padding = True
    actor.use_ulysses_sp = False
    actor.ulysses_sequence_parallel_size = 1
    actor.use_fused_kernels = False
    actor.use_dynamic_bsz = False
    actor.gradient_accumulation = 1
    actor.device_name = "cpu"
    actor.param_dtype = torch.float32
    actor.compute_entropy_from_logits = lambda logits: torch.zeros(logits.shape[:-1])

    input_ids = torch.tensor(
        [
            [1, 99, 99, 99, 99, 7, 8, 10, 11, 0, 0],
            [1, 99, 99, 99, 99, 99, 99, 7, 8, 10, 11],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    position_ids = torch.arange(input_ids.shape[1]).expand(2, -1)
    responses = torch.tensor([[10, 11], [10, 11]])
    multi_modal_inputs = [
        {"pixel_values": torch.randn(16, 1536), "image_grid_thw": torch.tensor([[1, 4, 4]])},
        {"pixel_values": torch.randn(24, 1536), "image_grid_thw": torch.tensor([[1, 4, 6]])},
    ]
    metadata = [
        {"bbox": [0, 0, 16, 16], "image_width": 32, "image_height": 32, "sample_id": "first"},
        {"bbox": [0, 0, 16, 16], "image_width": 48, "image_height": 32, "sample_id": "second"},
    ]
    student_batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "response_start_idx": torch.tensor([7, 9]),
        "multi_modal_inputs": multi_modal_inputs,
        "teacher_visual_token_selection_metadata": metadata,
    }

    student_output = actor._forward_micro_batch(
        student_batch,
        temperature=1.0,
        return_visual_token_selection=True,
    )
    state = student_output["visual_token_selection"]
    assert "image_embeds" not in student_output
    assert state["token_counts"] == [4, 6]
    assert [int(mask.sum()) for mask in state["keep_masks"]] == [2, 2]

    teacher_batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_start_idx": torch.tensor([7, 9]),
        "responses": responses,
        "multi_modal_inputs": [dict(inputs) for inputs in multi_modal_inputs],
    }
    finalized_teacher_batch, _ = actor._finalize_teacher_visual_token_mask(teacher_batch, state)
    assert finalized_teacher_batch["attention_mask"].sum(dim=1).tolist() == [7, 7]
    assert all(
        torch.equal(inputs["image_token_keep_mask"], keep_mask)
        for inputs, keep_mask in zip(
            finalized_teacher_batch["multi_modal_inputs"], state["keep_masks"], strict=True
        )
    )

    teacher_output = actor._forward_micro_batch(
        finalized_teacher_batch,
        temperature=1.0,
        module=model,
    )
    assert teacher_output["log_probs"].shape == (2, 2)
