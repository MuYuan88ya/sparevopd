from types import SimpleNamespace

import pytest
import torch

from verl.utils.teacher_visual_token_mask import (
    VisualTokenSelectionResult,
    create_visual_token_selection_strategy,
)
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
    actor.teacher_visual_image_token_id = 99
    actor.teacher_visual_spatial_merge_size = 2
    return actor


def make_inputs():
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


def execute_actor_session(actor, metadata, attention_mask):
    session = actor._build_visual_token_selection_session(metadata, attention_mask)
    image_mask = torch.zeros((1, int(attention_mask.sum())), dtype=torch.bool)
    first_length = int(attention_mask[0].sum())
    image_mask[0, 1:5] = True
    image_mask[0, first_length + 1 : first_length + 7] = True
    session.execute(
        image_embeds=torch.randn(10, 8),
        inputs_embeds=torch.randn(1, int(attention_mask.sum()), 8),
        image_mask=image_mask,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 6]]),
    )
    return session.result


def test_teacher_consumes_session_result_without_mutating_student_inputs():
    actor = make_actor()
    multi_modal_inputs, selection_metadata, teacher_inputs = make_inputs()
    result = execute_actor_session(actor, selection_metadata, teacher_inputs["attention_mask"])
    finalized, metrics = actor._finalize_teacher_visual_token_mask(teacher_inputs, result)

    assert [mask.numel() for mask in result.keep_masks] == [4, 6]
    assert [int(mask.sum()) for mask in result.keep_masks] == [2, 5]
    assert all("image_token_keep_mask" not in inputs for inputs in multi_modal_inputs)
    assert all(
        torch.equal(final_inputs["image_token_keep_mask"], result_mask)
        for final_inputs, result_mask in zip(finalized["multi_modal_inputs"], result.keep_masks, strict=True)
    )
    for sample_idx, keep_mask in enumerate(result.keep_masks):
        active_image_positions = (
            teacher_inputs["input_ids"][sample_idx].eq(99)
            & finalized["attention_mask"][sample_idx].bool()
        )
        assert int(active_image_positions.sum()) == int(keep_mask.sum())
    assert torch.equal(finalized["attention_mask"][0, 6:], teacher_inputs["attention_mask"][0, 6:])
    assert torch.equal(finalized["attention_mask"][1, 8:], teacher_inputs["attention_mask"][1, 8:])
    assert metrics["actor/teacher_kept_visual_tokens"] == 7.0
    assert metrics["actor/teacher_llm_effective_tokens"] == float(finalized["attention_mask"].sum())


def test_teacher_rejects_result_batch_mismatch():
    actor = make_actor()
    _, _, teacher_inputs = make_inputs()
    invalid_result = VisualTokenSelectionResult(
        keep_masks=(torch.ones(4, dtype=torch.bool),),
        target_keep_masks=(torch.ones(4, dtype=torch.bool),),
        sample_ids=("first",),
        metrics={},
    )
    with pytest.raises(ValueError, match="one entry per teacher sample"):
        actor._finalize_teacher_visual_token_mask(teacher_inputs, invalid_result)


def test_actor_session_builder_rejects_non_matrix_attention_mask():
    actor = make_actor()
    with pytest.raises(ValueError, match="attention_mask must have shape"):
        actor._build_visual_token_selection_session(
            [{"bbox": [0, 0, 1, 1], "image_width": 2, "image_height": 2, "sample_id": 1}],
            torch.ones(7, dtype=torch.long),
        )


def configure_full_actor(model, *, background_keep_ratio):
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = {
        "calculate_sum_pi_squared": False,
        "sum_pi_squared_checkpointing": False,
        "entropy_checkpointing": False,
    }
    actor.actor_module = model
    actor.teacher_visual_token_selection_strategy = create_visual_token_selection_strategy(
        "fixed_stratified", {"background_keep_ratio": background_keep_ratio, "seed": 0}
    )
    actor.teacher_visual_image_token_id = 99
    actor.teacher_visual_spatial_merge_size = 2
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
    return actor


def make_full_flow_batch():
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
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "response_start_idx": torch.tensor([7, 9]),
        "multi_modal_inputs": multi_modal_inputs,
        "teacher_visual_token_selection_metadata": metadata,
    }
    return batch, metadata


@pytest.fixture
def patch_qwen_and_unpadding(monkeypatch):
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


def test_full_student_session_to_teacher_flow(patch_qwen_and_unpadding):
    model = make_tiny_qwen3_5().eval()
    actor = configure_full_actor(model, background_keep_ratio=0.1)
    student_batch, metadata = make_full_flow_batch()
    dense_student_output = actor._forward_micro_batch(student_batch, temperature=1.0)
    session = actor._build_visual_token_selection_session(metadata, student_batch["attention_mask"])

    student_output = actor._forward_micro_batch(
        student_batch,
        temperature=1.0,
        visual_token_selection_session=session,
    )
    result = session.result
    assert set(student_output) == {"log_probs"}
    assert torch.allclose(student_output["log_probs"], dense_student_output["log_probs"])
    assert [mask.numel() for mask in result.keep_masks] == [4, 6]
    assert [int(mask.sum()) for mask in result.keep_masks] == [2, 2]

    teacher_batch = {
        "input_ids": student_batch["input_ids"],
        "attention_mask": student_batch["attention_mask"],
        "position_ids": student_batch["position_ids"],
        "response_start_idx": student_batch["response_start_idx"],
        "responses": student_batch["responses"],
        "multi_modal_inputs": [dict(inputs) for inputs in student_batch["multi_modal_inputs"]],
    }
    finalized_teacher_batch, _ = actor._finalize_teacher_visual_token_mask(teacher_batch, result)
    teacher_output = actor._forward_micro_batch(finalized_teacher_batch, temperature=1.0, module=model)
    assert teacher_output["log_probs"].shape == (2, 2)


def test_full_keep_teacher_matches_dense_teacher_logits(patch_qwen_and_unpadding):
    model = make_tiny_qwen3_5().eval()
    actor = configure_full_actor(model, background_keep_ratio=1.0)
    batch, metadata = make_full_flow_batch()
    session = actor._build_visual_token_selection_session(metadata, batch["attention_mask"])
    actor._forward_micro_batch(batch, temperature=1.0, visual_token_selection_session=session)

    dense_teacher_batch = {
        key: value
        for key, value in batch.items()
        if key != "teacher_visual_token_selection_metadata"
    }
    dense_output = actor._forward_micro_batch(dense_teacher_batch, temperature=1.0, module=model)
    sparse_teacher_batch = {
        **dense_teacher_batch,
        "multi_modal_inputs": [dict(inputs) for inputs in batch["multi_modal_inputs"]],
    }
    sparse_teacher_batch, _ = actor._finalize_teacher_visual_token_mask(
        sparse_teacher_batch,
        session.result,
    )
    sparse_output = actor._forward_micro_batch(sparse_teacher_batch, temperature=1.0, module=model)
    assert torch.allclose(sparse_output["log_probs"], dense_output["log_probs"])
