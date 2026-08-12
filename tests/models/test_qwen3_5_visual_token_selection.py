import pytest
import torch

try:
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
except ImportError:  # pragma: no cover - depends on the local transformers build
    pytest.skip("Qwen3.5 is unavailable in this transformers build", allow_module_level=True)

from verl.models.transformers.qwen3_5 import _get_input_embeds
from verl.utils.teacher_visual_token_mask import (
    VisualTokenSelectionSession,
    VisualTokenSelectionStrategy,
    create_visual_token_selection_strategy,
)


def make_tiny_qwen3_5():
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
    model = Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=99,
            video_token_id=100,
        )
    ).model
    model.eval()
    return model


def make_session(strategy=None):
    return VisualTokenSelectionSession(
        strategy=strategy
        or create_visual_token_selection_strategy(
            "fixed_stratified",
            {"background_keep_ratio": 0.1},
        ),
        metadata=({"bbox": [0, 0, 16, 16], "image_width": 32, "image_height": 32, "sample_id": "sample"},),
        valid_sequence_lengths=(7,),
        spatial_merge_size=2,
    )


def test_student_session_selects_mask_without_changing_dense_inputs_or_output_shape():
    model = make_tiny_qwen3_5()
    input_ids = torch.tensor([[1, 99, 99, 99, 99, 2, 3]])
    pixel_values = torch.randn(16, 3 * 2 * 16 * 16)
    image_grid_thw = torch.tensor([[1, 4, 4]])
    session = make_session()

    with torch.no_grad():
        dense = _get_input_embeds(
            model,
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        selected = _get_input_embeds(
            model,
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            visual_token_selection_session=session,
        )

    assert set(selected) == {"inputs_embeds", "attention_mask"}
    assert torch.allclose(selected["inputs_embeds"], dense["inputs_embeds"])
    assert session.result.keep_masks[0].dtype == torch.bool
    assert int(session.result.keep_masks[0].sum()) == 2
    assert not session.result.keep_masks[0].requires_grad


def test_strategy_sees_per_sample_dense_multimodal_embeddings_and_image_positions():
    model = make_tiny_qwen3_5()
    observed = {}

    class RecordingStrategy(VisualTokenSelectionStrategy):
        def select(self, sample):
            observed.update(
                inputs_embeds=sample.inputs_embeds,
                image_embeds=sample.image_embeds,
                image_mask=sample.image_mask,
            )
            return sample.target_keep_mask.clone()

    input_ids = torch.tensor([[1, 99, 99, 99, 99, 2, 3]])
    pixel_values = torch.randn(16, 3 * 2 * 16 * 16)
    session = make_session(strategy=RecordingStrategy())

    with torch.no_grad():
        _get_input_embeds(
            model,
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            visual_token_selection_session=session,
        )

    assert observed["inputs_embeds"].shape == (7, 32)
    assert observed["image_embeds"].shape == (4, 32)
    assert observed["image_mask"].shape == (7,)
    assert torch.equal(observed["image_mask"], input_ids.eq(99).squeeze(0))
    assert torch.allclose(observed["inputs_embeds"][observed["image_mask"]], observed["image_embeds"])


def test_teacher_filters_dense_visual_features_after_attention_unpadding():
    model = make_tiny_qwen3_5()
    pixel_values = torch.randn(16, 3 * 2 * 16 * 16)
    image_grid_thw = torch.tensor([[1, 4, 4]])
    keep_mask = torch.tensor([True, True, False, False])
    teacher_input_ids = torch.tensor([[1, 99, 99, 2, 3]])

    with torch.no_grad():
        dense_output = _get_input_embeds(
            model,
            torch.tensor([[1, 99, 99, 99, 99, 2, 3]]),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        all_kept_output = _get_input_embeds(
            model,
            torch.tensor([[1, 99, 99, 99, 99, 2, 3]]),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_keep_mask=torch.ones(4, dtype=torch.bool),
        )
        sparse_output = _get_input_embeds(
            model,
            teacher_input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_keep_mask=keep_mask,
        )

    assert torch.allclose(dense_output["inputs_embeds"], all_kept_output["inputs_embeds"])
    assert sparse_output["inputs_embeds"].shape[:2] == teacher_input_ids.shape

    with pytest.raises(ValueError, match="active placeholders"):
        _get_input_embeds(
            model,
            torch.tensor([[1, 99, 99, 99, 2, 3]]),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_keep_mask=keep_mask,
        )


def test_student_session_and_teacher_keep_mask_are_mutually_exclusive():
    model = make_tiny_qwen3_5()
    with pytest.raises(ValueError, match="cannot be enabled together"):
        _get_input_embeds(
            model,
            torch.tensor([[1, 99, 99, 99, 99, 2, 3]]),
            pixel_values=torch.randn(16, 3 * 2 * 16 * 16),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            image_token_keep_mask=torch.ones(4, dtype=torch.bool),
            visual_token_selection_session=make_session(),
        )


def test_student_session_rejects_video_inputs():
    model = make_tiny_qwen3_5()
    with pytest.raises(ValueError, match="still images only"):
        _get_input_embeds(
            model,
            torch.tensor([[1, 99, 99, 99, 99, 2, 3]]),
            pixel_values=torch.randn(16, 3 * 2 * 16 * 16),
            pixel_values_videos=torch.randn(16, 3 * 2 * 16 * 16),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            video_grid_thw=torch.tensor([[1, 4, 4]]),
            visual_token_selection_session=make_session(),
        )


def test_student_session_requires_image_grid():
    model = make_tiny_qwen3_5()
    with pytest.raises(ValueError, match="requires image_grid_thw"):
        _get_input_embeds(
            model,
            torch.tensor([[1, 99, 99, 99, 99, 2, 3]]),
            pixel_values=torch.randn(16, 3 * 2 * 16 * 16),
            visual_token_selection_session=make_session(),
        )
