import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import verl.utils.torch_functional as verl_F
from verl.utils.teacher_entropy_reweighting import (
    TeacherEntropyWeightAccumulator,
    global_valid_token_count,
    scale_gradients_,
)


def _distributed_entropy_worker(rank, world_size, init_method, result_queue):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        local_entropy = torch.tensor([[0.0, 2.0]]) if rank == 0 else torch.tensor([[1.0, 4.0, 8.0]])
        local_mask = torch.ones_like(local_entropy)
        token_count = global_valid_token_count(local_mask)
        accumulator = TeacherEntropyWeightAccumulator(beta=0.5, global_token_count=token_count)
        confidence = accumulator.add(local_entropy, local_mask)
        accumulator.add_weighted_loss_sum(confidence.sum())
        scale, metrics = accumulator.finalize()
        result_queue.put((rank, token_count.item(), scale, metrics))
    finally:
        dist.destroy_process_group()


def test_full_vocab_entropy_matches_direct_softmax_computation():
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.0, 0.0, 0.0]], dtype=torch.float64)
    probabilities = torch.softmax(logits, dim=-1)
    expected = -(probabilities * torch.log(probabilities)).sum(dim=-1)

    actual = verl_F.entropy_from_logits(logits)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_two_micro_batches_match_direct_global_normalization_and_gradient():
    entropy_chunks = [torch.tensor([[0.2, 1.0]]), torch.tensor([[2.0, 4.0, 8.0]])]
    mask_chunks = [torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0, 0.0, 1.0]])]
    coefficient_chunks = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0, 5.0]])]
    full_mask = torch.cat([chunk.flatten() for chunk in mask_chunks])
    token_count = global_valid_token_count(full_mask)
    accumulator = TeacherEntropyWeightAccumulator(beta=0.7, global_token_count=token_count)

    parameter = torch.tensor(1.25, requires_grad=True)
    confidence_chunks = []
    for entropy, mask, coefficient in zip(entropy_chunks, mask_chunks, coefficient_chunks, strict=True):
        confidence = accumulator.add(entropy, mask)
        confidence_chunks.append(confidence)
        per_token_loss = coefficient * parameter.square()
        weighted_sum = (confidence * per_token_loss * mask).sum()
        accumulator.add_weighted_loss_sum(weighted_sum.detach())
        (weighted_sum / token_count).backward()

    normalization_scale, metrics = accumulator.finalize()
    scale_gradients_([parameter], normalization_scale)

    reference_parameter = torch.tensor(1.25, requires_grad=True)
    valid_entropy = torch.cat(
        [entropy[mask.bool()] for entropy, mask in zip(entropy_chunks, mask_chunks, strict=True)]
    )
    valid_coefficients = torch.cat(
        [coefficient[mask.bool()] for coefficient, mask in zip(coefficient_chunks, mask_chunks, strict=True)]
    )
    confidence = torch.exp(-0.7 * valid_entropy)
    normalized_weight = confidence / confidence.mean()
    reference_loss = (normalized_weight * valid_coefficients * reference_parameter.square()).mean()
    reference_loss.backward()

    assert torch.allclose(parameter.grad, reference_parameter.grad, atol=1e-6, rtol=1e-6)
    assert metrics["self_distillation/entropy_weight_normalization_scale"] == pytest.approx(
        normalization_scale
    )
    assert metrics["self_distillation/entropy_weighted_jsd_token_mean"] == pytest.approx(
        (confidence * valid_coefficients * 1.25**2).sum().item() / confidence.sum().item()
    )
    assert metrics["self_distillation/entropy_weight_effective_token_fraction"] <= 1.0


def test_beta_zero_produces_exact_unit_weights_and_unit_scale():
    entropy = torch.tensor([[0.0, 1.0, 10.0]])
    mask = torch.ones_like(entropy)
    token_count = global_valid_token_count(mask)
    accumulator = TeacherEntropyWeightAccumulator(beta=0.0, global_token_count=token_count)

    confidence = accumulator.add(entropy, mask)
    accumulator.add_weighted_loss_sum(torch.tensor(6.0))
    normalization_scale, metrics = accumulator.finalize()

    assert torch.equal(confidence, torch.ones_like(confidence))
    assert normalization_scale == 1.0
    assert metrics["self_distillation/entropy_weight_min"] == 1.0
    assert metrics["self_distillation/entropy_weight_max"] == 1.0
    assert metrics["self_distillation/entropy_weight_std"] == 0.0
    assert metrics["self_distillation/entropy_weight_effective_token_fraction"] == 1.0


def test_larger_beta_increases_relative_preference_for_lower_entropy():
    entropy = torch.tensor([[0.1, 1.0, 3.0]])
    mask = torch.ones_like(entropy)
    normalized_weights = []
    for beta in (0.5, 2.0):
        accumulator = TeacherEntropyWeightAccumulator(
            beta=beta,
            global_token_count=global_valid_token_count(mask),
        )
        confidence = accumulator.add(entropy, mask)
        accumulator.add_weighted_loss_sum(torch.tensor(0.0))
        scale, _ = accumulator.finalize()
        normalized_weights.append(confidence * scale)

    assert normalized_weights[1][0, 0] > normalized_weights[0][0, 0]
    assert normalized_weights[1][0, -1] < normalized_weights[0][0, -1]
    assert normalized_weights[1].mean().item() == pytest.approx(1.0)


def test_accumulator_rejects_invalid_statistics_and_lifecycle():
    mask = torch.ones((1, 2))
    with pytest.raises(ValueError, match="non-negative"):
        TeacherEntropyWeightAccumulator(beta=-1.0, global_token_count=global_valid_token_count(mask))

    accumulator = TeacherEntropyWeightAccumulator(beta=1.0, global_token_count=global_valid_token_count(mask))
    with pytest.raises(ValueError, match="non-finite"):
        accumulator.add(torch.tensor([[math.nan, 0.0]]), mask)

    accumulator = TeacherEntropyWeightAccumulator(beta=1.0, global_token_count=global_valid_token_count(mask))
    accumulator.add(torch.zeros_like(mask), mask)
    accumulator.add_weighted_loss_sum(0.0)
    accumulator.finalize()
    with pytest.raises(RuntimeError, match="only be called once"):
        accumulator.finalize()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_global_weights_are_independent_of_rank_partition(tmp_path):
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    init_method = f"file://{tmp_path / 'entropy_reweighting_init'}"
    mp.spawn(
        _distributed_entropy_worker,
        args=(2, init_method, result_queue),
        nprocs=2,
        join=True,
    )
    results = sorted((result_queue.get(), result_queue.get()))

    full_entropy = torch.tensor([0.0, 2.0, 1.0, 4.0, 8.0])
    confidence = torch.exp(-0.5 * full_entropy)
    expected_scale = full_entropy.numel() / confidence.sum().item()
    expected_std = (confidence * expected_scale).std(unbiased=False).item()
    for rank, token_count, scale, metrics in results:
        assert rank in (0, 1)
        assert token_count == 5
        assert scale == pytest.approx(expected_scale)
        assert metrics["self_distillation/teacher_entropy_mean"] == pytest.approx(
            full_entropy.mean().item()
        )
        assert metrics["self_distillation/entropy_weight_std"] == pytest.approx(expected_std)
        assert metrics["self_distillation/entropy_weighted_jsd_token_mean"] == pytest.approx(1.0)
