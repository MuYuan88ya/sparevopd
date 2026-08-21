import math
from dataclasses import fields

import pytest
import torch

from verl.utils.teacher_visual_token_mask import (
    VisualTokenSelectionResult,
    VisualTokenSelectionSample,
    VisualTokenSelectionSession,
    VisualTokenSelectionStrategy,
    apply_image_token_keep_mask,
    build_bbox_target_keep_mask,
    create_visual_token_selection_strategy,
    register_visual_token_selection_strategy,
)


def make_session(
    *,
    strategy=None,
    sample_id=1234,
    bbox=(20, 30, 40, 50),
    valid_sequence_lengths=(104,),
    contrastive=False,
    contrastive_negative_seed=0,
    global_step=0,
):
    return VisualTokenSelectionSession(
        strategy=strategy or create_visual_token_selection_strategy("fixed_stratified"),
        metadata=(
            {
                "bbox": bbox,
                "image_width": 100,
                "image_height": 100,
                "sample_id": sample_id,
            },
        ),
        valid_sequence_lengths=valid_sequence_lengths,
        spatial_merge_size=2,
        contrastive_negative_mode="complement-random-equal" if contrastive else None,
        contrastive_negative_seed=contrastive_negative_seed,
        global_step=global_step if contrastive else None,
    )


def execute_single(session):
    image_mask = torch.zeros((1, 104), dtype=torch.bool)
    image_mask[0, :100] = True
    session.execute(
        image_embeds=torch.randn(100, 8),
        inputs_embeds=torch.randn(1, 104, 8),
        image_mask=image_mask,
        image_grid_thw=torch.tensor([[1, 20, 20]]),
    )
    return session.result


def test_strategy_registry_factory_and_duplicate_registration():
    class TestStrategy(VisualTokenSelectionStrategy):
        def select(self, sample):
            return sample.target_keep_mask.clone()

    registered = register_visual_token_selection_strategy("test_session_strategy")(TestStrategy)
    assert registered is TestStrategy
    assert isinstance(create_visual_token_selection_strategy("test_session_strategy"), TestStrategy)

    with pytest.raises(ValueError, match="already registered"):
        register_visual_token_selection_strategy("test_session_strategy")(TestStrategy)


def test_factory_rejects_unknown_strategy_and_parameters():
    with pytest.raises(ValueError, match="Unknown visual token selection strategy"):
        create_visual_token_selection_strategy("missing")
    with pytest.raises(ValueError, match="Invalid parameters"):
        create_visual_token_selection_strategy("fixed_stratified", {"unknown": 1})
    with pytest.raises(ValueError, match="background_keep_ratio"):
        create_visual_token_selection_strategy("fixed_stratified", {"background_keep_ratio": 1.1})


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


@pytest.mark.parametrize("bbox", [[10, 10, 10, 20], [-1, 0, 10, 10], [0, 0, 101, 10]])
def test_invalid_bbox_is_rejected(bbox):
    with pytest.raises(ValueError, match="bbox"):
        execute_single(make_session(bbox=bbox))


def test_fixed_stratified_preserves_exact_pre_refactor_mask_positions():
    strategy = create_visual_token_selection_strategy(
        "fixed_stratified",
        {"background_keep_ratio": 0.1, "seed": 17},
    )
    result = execute_single(make_session(strategy=strategy, sample_id=1234))
    assert result.keep_masks[0].nonzero().flatten().tolist() == [
        3,
        19,
        21,
        24,
        32,
        33,
        39,
        41,
        42,
        43,
        55,
        75,
        87,
        90,
    ]


@pytest.mark.parametrize("ratio,expected_background", [(0.0, 0), (0.1, 10), (1.0, 96)])
def test_fixed_stratified_keeps_configured_background_and_all_targets(ratio, expected_background):
    strategy = create_visual_token_selection_strategy(
        "fixed_stratified",
        {"background_keep_ratio": ratio, "seed": 3},
    )
    result = execute_single(make_session(strategy=strategy))
    keep_mask = result.keep_masks[0]
    target_mask = result.target_keep_masks[0]
    assert int((keep_mask.cpu() & ~target_mask).sum()) == expected_background
    assert int(keep_mask.sum()) == 4 + expected_background
    assert torch.all(keep_mask.cpu()[target_mask])


def test_fixed_stratified_rounds_nonzero_background_ratio_up():
    session = VisualTokenSelectionSession(
        strategy=create_visual_token_selection_strategy(
            "fixed_stratified",
            {"background_keep_ratio": 0.1},
        ),
        metadata=({"bbox": [0, 0, 1, 1], "image_width": 2, "image_height": 2, "sample_id": "ceil"},),
        valid_sequence_lengths=(6,),
        spatial_merge_size=2,
    )
    image_mask = torch.tensor([[False, True, True, True, True, False]])
    session.execute(
        image_embeds=torch.randn(4, 8),
        inputs_embeds=torch.randn(1, 6, 8),
        image_mask=image_mask,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
    )
    result = session.result
    assert int((result.keep_masks[0].cpu() & ~result.target_keep_masks[0]).sum()) == math.ceil(3 * 0.1)


@pytest.mark.parametrize(
    "seed,sample_id,expected_positions",
    [
        (17, 1234, [3, 19, 21, 24, 32, 33, 39, 41, 42, 43, 55, 75, 87, 90]),
        (18, 1234, [3, 14, 16, 32, 33, 36, 42, 43, 73, 75, 77, 78, 81, 89]),
        (17, 4321, [5, 11, 27, 32, 33, 38, 42, 43, 55, 61, 66, 69, 81, 97]),
    ],
)
def test_seed_and_sample_id_match_exact_pre_refactor_positions(seed, sample_id, expected_positions):
    result = execute_single(
        make_session(
            strategy=create_visual_token_selection_strategy(
                "fixed_stratified", {"background_keep_ratio": 0.1, "seed": seed}
            ),
            sample_id=sample_id,
        )
    )
    assert result.keep_masks[0].nonzero().flatten().tolist() == expected_positions


def test_batch_session_splits_dense_and_multimodal_context_per_sample():
    observed = []

    class RecordingStrategy(VisualTokenSelectionStrategy):
        def select(self, sample):
            observed.append(sample)
            return sample.target_keep_mask.clone()

    session = VisualTokenSelectionSession(
        strategy=RecordingStrategy(),
        metadata=(
            {"bbox": [0, 0, 2, 2], "image_width": 4, "image_height": 4, "sample_id": "first"},
            {"bbox": [0, 0, 3, 4], "image_width": 6, "image_height": 4, "sample_id": "second"},
        ),
        valid_sequence_lengths=(7, 9),
        spatial_merge_size=2,
    )
    image_mask = torch.zeros((1, 16), dtype=torch.bool)
    image_mask[0, 1:5] = True
    image_mask[0, 8:14] = True
    session.execute(
        image_embeds=torch.randn(10, 8),
        inputs_embeds=torch.randn(1, 16, 8),
        image_mask=image_mask,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 6]]),
    )

    assert [sample.image_embeds.shape for sample in observed] == [(4, 8), (6, 8)]
    assert [sample.inputs_embeds.shape for sample in observed] == [(7, 8), (9, 8)]
    assert [int(sample.image_mask.sum()) for sample in observed] == [4, 6]
    assert [sample.sample_id for sample in observed] == ["first", "second"]
    assert [sample.grid_width for sample in observed] == [2, 3]
    assert [int(mask.sum()) for mask in session.result.keep_masks] == [1, 4]


def test_session_is_single_use_and_result_retains_only_compact_fields():
    session = make_session()
    with pytest.raises(RuntimeError, match="has not been executed"):
        _ = session.result
    result = execute_single(session)
    assert [field.name for field in fields(VisualTokenSelectionResult)] == [
        "keep_masks",
        "target_keep_masks",
        "sample_ids",
        "metrics",
        "negative_keep_masks",
    ]
    assert all(mask.numel() == 100 for mask in (*result.keep_masks, *result.target_keep_masks))
    with pytest.raises(RuntimeError, match="only be executed once"):
        session.execute(
            image_embeds=torch.randn(100, 8),
            inputs_embeds=torch.randn(1, 104, 8),
            image_mask=torch.ones((1, 104), dtype=torch.bool),
            image_grid_thw=torch.tensor([[1, 20, 20]]),
        )


def test_contrastive_negative_mask_is_disjoint_equal_size_and_excludes_target():
    result = execute_single(
        make_session(
            strategy=create_visual_token_selection_strategy(
                "fixed_stratified", {"background_keep_ratio": 0.3, "seed": 7}
            ),
            contrastive=True,
            contrastive_negative_seed=11,
            global_step=23,
        )
    )
    positive_mask = result.keep_masks[0]
    negative_mask = result.negative_keep_masks[0]
    target_mask = result.target_keep_masks[0].to(negative_mask.device)

    assert int(positive_mask.sum()) == int(negative_mask.sum())
    assert not torch.any(positive_mask & negative_mask)
    assert not torch.any(target_mask & negative_mask)
    assert torch.all(negative_mask <= ~positive_mask)


def test_contrastive_negative_mask_is_stable_within_step_and_changes_across_context():
    def select(*, sample_id="sample", global_step=10, seed=3):
        return execute_single(
            make_session(
                sample_id=sample_id,
                contrastive=True,
                contrastive_negative_seed=seed,
                global_step=global_step,
            )
        ).negative_keep_masks[0]

    reference = select()
    assert torch.equal(reference, select())
    assert not torch.equal(reference, select(global_step=11))
    assert not torch.equal(reference, select(sample_id="other"))
    assert not torch.equal(reference, select(seed=4))


def test_contrastive_session_requires_global_step():
    with pytest.raises(ValueError, match="requires a global_step"):
        make_session(contrastive=True, global_step=None)


def test_contrastive_negative_mask_rejects_insufficient_complement_with_counts():
    session = make_session(
        strategy=create_visual_token_selection_strategy(
            "fixed_stratified", {"background_keep_ratio": 0.3, "seed": 0}
        ),
        bbox=(0, 0, 80, 80),
        contrastive=True,
        global_step=1,
    )
    with pytest.raises(
        ValueError,
        match=r"sample_id=1234.*total_tokens=100.*bbox_tokens=64.*positive_tokens=75.*complement_tokens=25",
    ):
        execute_single(session)


class InvalidOutputStrategy(VisualTokenSelectionStrategy):
    def __init__(self, output):
        self.output = output

    def select(self, sample: VisualTokenSelectionSample):
        return self.output(sample) if callable(self.output) else self.output


@pytest.mark.parametrize(
    "output,error",
    [
        (None, "expected torch.Tensor"),
        (lambda sample: torch.ones(sample.target_keep_mask.shape), "rank-1 bool"),
        (lambda sample: torch.ones(99, dtype=torch.bool), "must match"),
        (lambda sample: torch.zeros_like(sample.target_keep_mask), "retain every target"),
    ],
)
def test_strategy_output_contract_is_validated(output, error):
    with pytest.raises((TypeError, ValueError), match=error):
        execute_single(make_session(strategy=InvalidOutputStrategy(output)))


@pytest.mark.parametrize(
    "override,error",
    [
        ({"image_embeds": torch.randn(99, 8)}, "dense image embeddings"),
        (
            {
                "inputs_embeds": torch.randn(1, 103, 8),
                "image_mask": torch.cat([torch.ones(100), torch.zeros(3)]).bool().unsqueeze(0),
            },
            "valid sequence lengths",
        ),
        ({"image_grid_thw": torch.tensor([[2, 20, 20]])}, "still images"),
        ({"image_grid_thw": torch.tensor([[1.0, 20.0, 20.0]])}, "integer dtype"),
        (
            {"image_grid_thw": torch.tensor([[1, 20, 20], [1, 20, 20]])},
            "exactly one image per sample",
        ),
        (
            {"image_mask": torch.cat([torch.ones(99), torch.zeros(5)]).bool().unsqueeze(0)},
            "image_mask and dense image embeddings",
        ),
    ],
)
def test_session_rejects_misaligned_dense_context(override, error):
    args = {
        "image_embeds": torch.randn(100, 8),
        "inputs_embeds": torch.randn(1, 104, 8),
        "image_mask": torch.cat([torch.ones(100), torch.zeros(4)]).bool().unsqueeze(0),
        "image_grid_thw": torch.tensor([[1, 20, 20]]),
    }
    args.update(override)
    with pytest.raises(ValueError, match=error):
        make_session().execute(**args)


def test_missing_bbox_metadata_is_rejected_inside_session():
    session = VisualTokenSelectionSession(
        strategy=create_visual_token_selection_strategy("fixed_stratified"),
        metadata=({"image_width": 100, "image_height": 100, "sample_id": 1},),
        valid_sequence_lengths=(104,),
        spatial_merge_size=2,
    )
    with pytest.raises(KeyError, match="bbox"):
        execute_single(session)


def test_missing_sample_id_is_rejected_inside_session():
    session = make_session(sample_id=None)
    with pytest.raises(ValueError, match="sample_id must not be None"):
        execute_single(session)


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
