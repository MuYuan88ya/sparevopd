import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).parents[2] / "verl" / "utils" / "teacher_visual_token_mask.py"
SPEC = importlib.util.spec_from_file_location("teacher_visual_token_mask", MODULE_PATH)
teacher_visual_token_mask = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = teacher_visual_token_mask
SPEC.loader.exec_module(teacher_visual_token_mask)

apply_image_token_keep_mask = teacher_visual_token_mask.apply_image_token_keep_mask
build_bbox_sparse_keep_mask = teacher_visual_token_mask.build_bbox_sparse_keep_mask


def test_bbox_tokens_and_exact_background_ratio_are_kept():
    selection = build_bbox_sparse_keep_mask(
        bbox=[20, 30, 40, 50],
        image_width=100,
        image_height=100,
        grid_height=10,
        grid_width=10,
        background_keep_ratio=0.1,
        sample_id=7,
        seed=3,
    )

    expected_target_indices = {32, 33, 42, 43}
    assert all(selection.keep_mask[index] for index in expected_target_indices)
    assert selection.target_token_count == 4
    assert selection.background_token_count == 96
    assert selection.kept_background_token_count == math.ceil(96 * 0.1)
    assert selection.keep_mask.sum().item() == 14


def test_tiny_boundary_bbox_keeps_at_least_one_token():
    selection = build_bbox_sparse_keep_mask(
        bbox=[99, 99, 100, 100],
        image_width=100,
        image_height=100,
        grid_height=10,
        grid_width=10,
        background_keep_ratio=0.0,
        sample_id="tiny",
    )

    assert selection.target_token_count == 1
    assert selection.keep_mask.nonzero(as_tuple=False).flatten().tolist() == [99]


def test_fixed_stratified_sampling_is_reproducible():
    kwargs = dict(
        bbox=[20, 30, 40, 50],
        image_width=100,
        image_height=100,
        grid_height=20,
        grid_width=20,
        background_keep_ratio=0.1,
        sample_id=1234,
        seed=17,
    )

    first = build_bbox_sparse_keep_mask(**kwargs)
    second = build_bbox_sparse_keep_mask(**kwargs)
    assert torch.equal(first.keep_mask, second.keep_mask)


@pytest.mark.parametrize("ratio,expected", [(0.0, 4), (1.0, 100)])
def test_background_ratio_extremes(ratio, expected):
    selection = build_bbox_sparse_keep_mask(
        bbox=[20, 30, 40, 50],
        image_width=100,
        image_height=100,
        grid_height=10,
        grid_width=10,
        background_keep_ratio=ratio,
        sample_id=0,
    )
    assert selection.keep_mask.sum().item() == expected


@pytest.mark.parametrize(
    "bbox",
    [
        [10, 10, 10, 20],
        [-1, 0, 10, 10],
        [0, 0, 101, 10],
    ],
)
def test_invalid_bbox_is_rejected(bbox):
    with pytest.raises(ValueError, match="bbox"):
        build_bbox_sparse_keep_mask(
            bbox=bbox,
            image_width=100,
            image_height=100,
            grid_height=10,
            grid_width=10,
            background_keep_ratio=0.1,
            sample_id=0,
        )


def test_attention_mask_only_disables_discarded_image_tokens():
    input_ids = torch.tensor([1, 99, 99, 99, 2, 3])
    attention_mask = torch.ones_like(input_ids)
    keep_mask = torch.tensor([True, False, True])

    teacher_attention_mask = apply_image_token_keep_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_token_id=99,
        image_token_keep_mask=keep_mask,
    )

    assert teacher_attention_mask.tolist() == [1, 1, 0, 1, 1, 1]
    assert attention_mask.tolist() == [1, 1, 1, 1, 1, 1]


def test_all_image_tokens_kept_is_identical_to_dense_attention_mask():
    input_ids = torch.tensor([0, 99, 99, 1, 2])
    attention_mask = torch.tensor([0, 1, 1, 1, 1])

    teacher_attention_mask = apply_image_token_keep_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_token_id=99,
        image_token_keep_mask=torch.ones(2, dtype=torch.bool),
    )

    assert torch.equal(teacher_attention_mask, attention_mask)
    assert teacher_attention_mask.data_ptr() != attention_mask.data_ptr()


def test_attention_mask_rejects_feature_placeholder_mismatch():
    with pytest.raises(ValueError, match="placeholders"):
        apply_image_token_keep_mask(
            input_ids=torch.tensor([99, 99]),
            attention_mask=torch.ones(2, dtype=torch.long),
            image_token_id=99,
            image_token_keep_mask=torch.tensor([True]),
        )
