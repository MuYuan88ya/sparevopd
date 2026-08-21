import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.utils.distillation_token_weight_metrics import DistillationTokenWeightMetricsAccumulator
from verl.utils.teacher_entropy_reweighting import global_valid_token_count


def _count(token_count):
    return torch.tensor(float(token_count), dtype=torch.float64)


def _distributed_metrics_worker(rank, world_size, init_method, result_queue):
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    try:
        weights = torch.tensor([[0.0, 1.0]]) if rank == 0 else torch.tensor([[2.0, 3.0, 4.0]])
        losses = weights + 1.0
        entropy = weights * 0.1
        mask = torch.ones_like(weights)
        token_count = global_valid_token_count(mask)
        accumulator = DistillationTokenWeightMetricsAccumulator(
            mode="teacher_entropy",
            global_token_count=token_count,
            sample_size=16,
            vocab_size=128,
        )
        accumulator.add(
            token_weights=weights,
            raw_per_token_loss=losses,
            loss_mask=mask,
            source_signal=entropy,
        )
        result_queue.put((rank, accumulator.finalize(normalization_scale=0.5)))
    finally:
        dist.destroy_process_group()


def test_entropy_weight_metrics_match_direct_global_statistics():
    raw_weights = torch.tensor([[1.0, 0.5, 0.25, 0.125]])
    raw_jsd = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    entropy = torch.tensor([[0.0, 0.5, 1.0, 2.0]])
    mask = torch.ones_like(raw_weights)
    scale = raw_weights.numel() / raw_weights.sum().item()
    accumulator = DistillationTokenWeightMetricsAccumulator(
        mode="teacher_entropy",
        global_token_count=_count(4),
        sample_size=16,
        vocab_size=128,
    )

    accumulator.add(
        token_weights=raw_weights,
        raw_per_token_loss=raw_jsd,
        loss_mask=mask,
        source_signal=entropy,
    )
    metrics = accumulator.finalize(normalization_scale=scale)

    weights = raw_weights.flatten().double() * scale
    losses = raw_jsd.flatten().double()
    expected_effective_fraction = weights.sum().square() / (weights.numel() * weights.square().sum())
    expected_correlation = torch.corrcoef(torch.stack((weights, losses)))[0, 1]
    expected_ratio = (weights * losses).sum() / losses.sum()
    assert metrics["self_distillation/token_weight_mean"] == pytest.approx(1.0)
    assert metrics["self_distillation/token_weight_std"] == pytest.approx(
        weights.std(unbiased=False).item()
    )
    assert metrics["self_distillation/token_weight_effective_token_fraction"] == pytest.approx(
        expected_effective_fraction.item()
    )
    assert metrics["self_distillation/token_weight_jsd_pearson"] == pytest.approx(
        expected_correlation.item()
    )
    assert metrics["self_distillation/weighted_to_raw_jsd_ratio"] == pytest.approx(
        expected_ratio.item()
    )
    assert metrics["self_distillation/teacher_entropy_std"] == pytest.approx(
        entropy.double().std(unbiased=False).item()
    )
    assert metrics["self_distillation/teacher_normalized_entropy_mean"] == pytest.approx(
        entropy.mean().item() / math.log(128)
    )


def test_visual_advantage_metrics_preserve_negative_margin_information():
    margin = torch.tensor([[-1.0, 0.2, 2.0]])
    weights = torch.relu(margin)
    raw_jsd = torch.tensor([[3.0, 2.0, 1.0]])
    positive_log_probs = torch.tensor([[-3.0, -1.0, -0.5]])
    negative_log_probs = positive_log_probs - margin
    accumulator = DistillationTokenWeightMetricsAccumulator(
        mode="visual_advantage",
        global_token_count=_count(3),
        sample_size=16,
    )

    accumulator.add(
        token_weights=weights,
        raw_per_token_loss=raw_jsd,
        loss_mask=torch.ones_like(weights),
        source_signal=margin,
        positive_teacher_log_probs=positive_log_probs,
        negative_teacher_log_probs=negative_log_probs,
    )
    metrics = accumulator.finalize()

    assert metrics["self_distillation/token_weight_zero_fraction"] == pytest.approx(1 / 3)
    assert metrics["self_distillation/positive_teacher_win_fraction"] == pytest.approx(2 / 3)
    assert metrics["self_distillation/visual_advantage_margin_min"] == -1.0
    assert metrics["self_distillation/visual_advantage_margin_mean"] == pytest.approx(
        margin.mean().item()
    )
    assert metrics["self_distillation/positive_teacher_log_prob_mean"] == pytest.approx(
        positive_log_probs.mean().item()
    )
    assert metrics["self_distillation/negative_teacher_log_prob_mean"] == pytest.approx(
        negative_log_probs.mean().item()
    )
    assert metrics["self_distillation/positive_negative_log_prob_gap_mean"] == pytest.approx(
        margin.mean().item()
    )


def test_metrics_ignore_padding_and_bound_quantile_sample_size():
    accumulator = DistillationTokenWeightMetricsAccumulator(
        mode="teacher_entropy",
        global_token_count=_count(3),
        sample_size=2,
        vocab_size=32,
    )
    accumulator.add(
        token_weights=torch.tensor([[1.0, 2.0, 999.0, 3.0]]),
        raw_per_token_loss=torch.tensor([[2.0, 3.0, 999.0, 4.0]]),
        loss_mask=torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        source_signal=torch.tensor([[0.1, 0.2, 999.0, 0.3]]),
    )

    metrics = accumulator.finalize()

    assert metrics["self_distillation/token_weight_mean"] == pytest.approx(2.0)
    assert metrics["self_distillation/token_weight_max"] == 3.0
    assert metrics["self_distillation/teacher_entropy_std"] == pytest.approx(
        torch.tensor([0.1, 0.2, 0.3]).double().std(unbiased=False).item()
    )


def test_metrics_validate_shapes_values_and_lifecycle():
    accumulator = DistillationTokenWeightMetricsAccumulator(
        mode="teacher_entropy",
        global_token_count=_count(1),
        vocab_size=16,
    )
    with pytest.raises(ValueError, match="must match"):
        accumulator.add(
            token_weights=torch.ones((1, 2)),
            raw_per_token_loss=torch.ones((1, 1)),
            loss_mask=torch.ones((1, 1)),
            source_signal=torch.ones((1, 1)),
        )
    with pytest.raises(ValueError, match="non-negative"):
        accumulator.add(
            token_weights=torch.tensor([[-1.0]]),
            raw_per_token_loss=torch.ones((1, 1)),
            loss_mask=torch.ones((1, 1)),
            source_signal=torch.ones((1, 1)),
        )

    accumulator.add(
        token_weights=torch.ones((1, 1)),
        raw_per_token_loss=torch.ones((1, 1)),
        loss_mask=torch.ones((1, 1)),
        source_signal=torch.ones((1, 1)),
    )
    accumulator.finalize()
    with pytest.raises(RuntimeError, match="only be called once"):
        accumulator.finalize()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_metrics_cover_the_complete_global_token_set(tmp_path):
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    init_method = f"file://{tmp_path / 'token_weight_metrics_init'}"
    mp.spawn(
        _distributed_metrics_worker,
        args=(2, init_method, result_queue),
        nprocs=2,
        join=True,
    )
    results = sorted((result_queue.get(), result_queue.get()))

    normalized_weights = torch.arange(5, dtype=torch.float64) * 0.5
    expected_effective_fraction = normalized_weights.sum().square() / (
        normalized_weights.numel() * normalized_weights.square().sum()
    )
    for rank, metrics in results:
        assert rank in (0, 1)
        assert metrics["self_distillation/token_weight_mean"] == 1.0
        assert metrics["self_distillation/token_weight_std"] == pytest.approx(
            normalized_weights.std(unbiased=False).item()
        )
        assert metrics["self_distillation/token_weight_effective_token_fraction"] == pytest.approx(
            expected_effective_fraction.item()
        )
        assert metrics["self_distillation/token_weight_p50"] == pytest.approx(1.0)
