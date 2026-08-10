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

VisualTokenSelectionInput = teacher_visual_token_mask.VisualTokenSelectionInput
VisualTokenSelectionOutput = teacher_visual_token_mask.VisualTokenSelectionOutput
VisualTokenSelectionRequest = teacher_visual_token_mask.VisualTokenSelectionRequest
VisualTokenSelectionStrategy = teacher_visual_token_mask.VisualTokenSelectionStrategy
apply_image_token_keep_mask = teacher_visual_token_mask.apply_image_token_keep_mask
build_bbox_target_keep_mask = teacher_visual_token_mask.build_bbox_target_keep_mask
build_visual_token_selection_state = teacher_visual_token_mask.build_visual_token_selection_state
build_visual_token_selection_state_from_keep_mask = (
    teacher_visual_token_mask.build_visual_token_selection_state_from_keep_mask
)
create_strategy = teacher_visual_token_mask.create_visual_token_selection_strategy
register_strategy = teacher_visual_token_mask.register_visual_token_selection_strategy
select_batched_visual_tokens = teacher_visual_token_mask.select_batched_visual_tokens
select_visual_tokens_from_request = teacher_visual_token_mask.select_visual_tokens_from_request
select_visual_tokens = teacher_visual_token_mask.select_visual_tokens


def make_selection_input(*, target_keep_mask=None, sample_id=7):
    if target_keep_mask is None:
        target_keep_mask = build_bbox_target_keep_mask(
            bbox=[20, 30, 40, 50],
            image_width=100,
            image_height=100,
            grid_height=10,
            grid_width=10,
        )
    return VisualTokenSelectionInput(
        image_embeds=torch.randn(target_keep_mask.numel(), 8),
        image_grid_thw=torch.tensor([1, 20, 20]),
        target_keep_mask=target_keep_mask,
        sample_id=sample_id,
    )


def test_strategy_registry_factory_and_duplicate_registration():
    class TestStrategy(VisualTokenSelectionStrategy):
        def select(self, selection_input):
            return VisualTokenSelectionOutput(keep_mask=selection_input.target_keep_mask.clone())

    registered = register_strategy("test_registry_strategy")(TestStrategy)
    assert registered is TestStrategy
    assert isinstance(create_strategy("test_registry_strategy"), TestStrategy)

    with pytest.raises(ValueError, match="already registered"):
        register_strategy("test_registry_strategy")(TestStrategy)


def test_batch_strategy_receives_dense_context_and_compact_image_mask():
    observed = {}

    class RecordingStrategy(VisualTokenSelectionStrategy):
        def select(self, selection_input):
            observed["inputs_embeds"] = selection_input.inputs_embeds
            observed["image_mask"] = selection_input.image_mask
            observed["image_grid_thw"] = selection_input.image_grid_thw
            observed["sample_ids"] = selection_input.sample_ids
            return VisualTokenSelectionOutput(keep_mask=selection_input.target_keep_mask.clone())

    target_keep_mask = torch.tensor(
        [True, False, False, False, False, True, False, False, False, False],
        dtype=torch.bool,
    )
    image_mask = torch.zeros((1, 18), dtype=torch.bool)
    image_mask[0, [1, 2, 3, 4, 8, 9, 10, 11, 12, 13]] = True
    request = VisualTokenSelectionRequest(
        strategy=RecordingStrategy(),
        target_keep_mask=target_keep_mask,
        sample_ids=("first", "second"),
        spatial_merge_size=2,
    )

    output = select_visual_tokens_from_request(
        request=request,
        image_embeds=torch.randn(10, 8),
        inputs_embeds=torch.randn(1, 18, 8),
        image_mask=image_mask,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 6]]),
    )

    assert torch.equal(output.keep_mask, target_keep_mask)
    assert observed["inputs_embeds"].shape == (1, 18, 8)
    assert observed["image_mask"].shape == (1, 18)
    assert observed["image_mask"].dtype == torch.bool
    assert observed["image_grid_thw"].shape == (2, 3)
    assert observed["sample_ids"] == ("first", "second")


def test_factory_rejects_unknown_strategy_and_parameters():
    with pytest.raises(ValueError, match="Unknown visual token selection strategy"):
        create_strategy("missing")
    with pytest.raises(ValueError, match="Invalid parameters"):
        create_strategy("fixed_stratified", {"unknown": 1})
    with pytest.raises(ValueError, match="background_keep_ratio"):
        create_strategy("fixed_stratified", {"background_keep_ratio": 1.1})


def test_bbox_mapping_keeps_intersecting_tokens():
    target_keep_mask = build_bbox_target_keep_mask(
        bbox=[20, 30, 40, 50],
        image_width=100,
        image_height=100,
        grid_height=10,
        grid_width=10,
    )
    assert target_keep_mask.nonzero(as_tuple=False).flatten().tolist() == [32, 33, 42, 43]


def test_tiny_boundary_bbox_keeps_at_least_one_token():
    target_keep_mask = build_bbox_target_keep_mask(
        bbox=[99, 99, 100, 100],
        image_width=100,
        image_height=100,
        grid_height=10,
        grid_width=10,
    )
    assert target_keep_mask.nonzero(as_tuple=False).flatten().tolist() == [99]


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
        build_bbox_target_keep_mask(
            bbox=bbox,
            image_width=100,
            image_height=100,
            grid_height=10,
            grid_width=10,
        )


@pytest.mark.parametrize("ratio,expected_background", [(0.0, 0), (0.1, 10), (1.0, 96)])
def test_fixed_stratified_keeps_exact_background_ratio_and_all_targets(ratio, expected_background):
    selection_input = make_selection_input()
    output = select_visual_tokens(
        create_strategy("fixed_stratified", {"background_keep_ratio": ratio, "seed": 3}),
        selection_input,
    )

    target_count = int(selection_input.target_keep_mask.sum().item())
    background_count = int((output.keep_mask & ~selection_input.target_keep_mask).sum().item())
    assert target_count == 4
    assert background_count == expected_background
    assert output.keep_mask.sum().item() == target_count + expected_background
    assert torch.all(output.keep_mask[selection_input.target_keep_mask])


def test_fixed_stratified_rounds_nonzero_background_ratio_up():
    target_keep_mask = torch.tensor([True, False, False, False])
    selection_input = VisualTokenSelectionInput(
        image_embeds=torch.randn(4, 8),
        image_grid_thw=torch.tensor([1, 4, 4]),
        target_keep_mask=target_keep_mask,
        sample_id="ceil",
    )
    output = select_visual_tokens(
        create_strategy("fixed_stratified", {"background_keep_ratio": 0.1}),
        selection_input,
    )
    assert int((output.keep_mask & ~target_keep_mask).sum().item()) == math.ceil(3 * 0.1)


def test_fixed_stratified_sampling_is_reproducible():
    strategy = create_strategy("fixed_stratified", {"background_keep_ratio": 0.1, "seed": 17})
    selection_input = make_selection_input(sample_id=1234)
    first = select_visual_tokens(strategy, selection_input)
    second = select_visual_tokens(strategy, selection_input)
    assert torch.equal(first.keep_mask, second.keep_mask)


class InvalidOutputStrategy(VisualTokenSelectionStrategy):
    def __init__(self, output):
        self.output = output

    def select(self, selection_input):
        return self.output


@pytest.mark.parametrize(
    "output,error",
    [
        (torch.ones(100, dtype=torch.bool), "expected VisualTokenSelectionOutput"),
        (VisualTokenSelectionOutput(torch.ones(100)), "rank-1 bool"),
        (VisualTokenSelectionOutput(torch.ones(99, dtype=torch.bool)), "must match"),
        (VisualTokenSelectionOutput(torch.zeros(100, dtype=torch.bool)), "retain every target"),
    ],
)
def test_strategy_output_contract_is_validated(output, error):
    with pytest.raises((TypeError, ValueError), match=error):
        select_visual_tokens(InvalidOutputStrategy(output), make_selection_input())


@pytest.mark.parametrize(
    "selection_input,error",
    [
        (
            VisualTokenSelectionInput(
                image_embeds=torch.randn(99, 8),
                image_grid_thw=torch.tensor([1, 20, 20]),
                target_keep_mask=torch.zeros(100, dtype=torch.bool),
                sample_id=0,
            ),
            "do not match",
        ),
        (
            VisualTokenSelectionInput(
                image_embeds=torch.randn(100, 8),
                image_grid_thw=torch.tensor([1, 20, 20]),
                target_keep_mask=torch.zeros(100),
                sample_id=0,
            ),
            "rank-1 bool",
        ),
        (
            VisualTokenSelectionInput(
                image_embeds=torch.randn(100, 8),
                image_grid_thw=torch.tensor([2, 20, 20]),
                target_keep_mask=torch.zeros(100, dtype=torch.bool),
                sample_id=0,
            ),
            "one static image",
        ),
        (
            VisualTokenSelectionInput(
                image_embeds=torch.ones(100, 8, dtype=torch.long),
                image_grid_thw=torch.tensor([1, 20, 20]),
                target_keep_mask=torch.zeros(100, dtype=torch.bool),
                sample_id=0,
            ),
            "floating-point",
        ),
        (
            VisualTokenSelectionInput(
                image_embeds=torch.randn(100, 8),
                image_grid_thw=torch.tensor([1.0, 20.0, 20.0]),
                target_keep_mask=torch.zeros(100, dtype=torch.bool),
                sample_id=0,
            ),
            "integer dtype",
        ),
    ],
)
def test_strategy_input_contract_is_validated(selection_input, error):
    with pytest.raises(ValueError, match=error):
        select_visual_tokens(create_strategy("fixed_stratified"), selection_input)


def test_batched_selection_supports_different_image_token_counts():
    target_masks = [
        torch.tensor([True, False, False, False]),
        torch.tensor([False, True, False, False, False, False]),
    ]
    outputs = select_batched_visual_tokens(
        strategy=create_strategy("fixed_stratified", {"background_keep_ratio": 0.0}),
        image_embeds=torch.randn(10, 16),
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 6]]),
        target_keep_masks=target_masks,
        sample_ids=["first", "second"],
    )
    assert len(outputs) == 2
    assert torch.equal(outputs[0].keep_mask, target_masks[0])
    assert torch.equal(outputs[1].keep_mask, target_masks[1])


def test_batched_selection_rejects_flattened_embedding_mismatch():
    with pytest.raises(ValueError, match="flattened image embeddings"):
        select_batched_visual_tokens(
            strategy=create_strategy("fixed_stratified"),
            image_embeds=torch.randn(9, 4),
            image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 6]]),
            target_keep_masks=[torch.zeros(4, dtype=torch.bool), torch.zeros(6, dtype=torch.bool)],
            sample_ids=[0, 1],
        )


def test_selection_state_contains_only_detached_per_sample_masks():
    target_masks = [
        torch.tensor([True, False, False, False]),
        torch.tensor([False, True, False, False, False, False]),
    ]
    state = build_visual_token_selection_state(
        strategy=create_strategy("fixed_stratified", {"background_keep_ratio": 0.0}),
        image_embeds=torch.randn(10, 16),
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 6]]),
        target_keep_masks=target_masks,
        sample_ids=["first", "second"],
    )

    assert set(state) == {"keep_masks", "target_keep_masks", "token_counts", "sample_ids", "metrics"}
    assert "image_embeds" not in state
    assert state["token_counts"] == [4, 6]
    assert state["sample_ids"] == ["first", "second"]
    for keep_mask, target_mask, original_target in zip(
        state["keep_masks"], state["target_keep_masks"], target_masks, strict=True
    ):
        assert keep_mask.dtype == torch.bool and keep_mask.ndim == 1
        assert target_mask.dtype == torch.bool and target_mask.ndim == 1
        assert not keep_mask.requires_grad and not target_mask.requires_grad
        assert torch.equal(target_mask, original_target)
        assert torch.all(target_mask.to(keep_mask.device) <= keep_mask)


def test_selection_state_can_be_built_from_model_keep_mask_without_embeddings():
    target_masks = [
        torch.tensor([True, False, False, False]),
        torch.tensor([False, True, False, False, False, False]),
    ]
    state = build_visual_token_selection_state_from_keep_mask(
        keep_mask=torch.tensor(
            [True, False, False, False, False, True, False, False, False, False],
            dtype=torch.bool,
        ),
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 6]]),
        target_keep_masks=target_masks,
        sample_ids=["first", "second"],
        spatial_merge_size=2,
    )
    assert state["token_counts"] == [4, 6]
    assert [int(mask.sum()) for mask in state["keep_masks"]] == [1, 1]
    assert "image_embeds" not in state


def test_selection_state_rejects_strategy_target_drop():
    class DropTargetStrategy(VisualTokenSelectionStrategy):
        def select(self, selection_input):
            return VisualTokenSelectionOutput(keep_mask=torch.zeros_like(selection_input.target_keep_mask))

    with pytest.raises(ValueError, match="retain every target"):
        build_visual_token_selection_state(
            strategy=DropTargetStrategy(),
            image_embeds=torch.randn(4, 8),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            target_keep_masks=[torch.tensor([True, False, False, False])],
            sample_ids=[0],
        )


def test_attention_mask_only_disables_discarded_image_tokens_before_unpadding():
    input_ids = torch.tensor([1, 99, 99, 99, 2, 3, 4])
    attention_mask = torch.ones_like(input_ids)
    keep_mask = torch.tensor([True, False, True])

    teacher_attention_mask = apply_image_token_keep_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_token_id=99,
        image_token_keep_mask=keep_mask,
    )

    assert teacher_attention_mask.tolist() == [1, 1, 0, 1, 1, 1, 1]
    assert input_ids[teacher_attention_mask.bool()].tolist() == [1, 99, 99, 2, 3, 4]
    assert attention_mask.tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert teacher_attention_mask[-3:].tolist() == [1, 1, 1]


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
