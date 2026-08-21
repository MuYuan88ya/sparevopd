from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import compute_self_distillation_loss
from verl.utils.distillation_token_weight_metrics import DistillationTokenWeightMetricsAccumulator


def _config():
    return SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=None,
        distillation_add_tail=False,
        alpha=0.0,
        is_clip=None,
    )


def test_entropy_confidence_uses_generic_detached_token_weight_path():
    student_log_probs = torch.log(
        torch.tensor([[[0.8, 0.2], [0.6, 0.4], [0.5, 0.5]]], requires_grad=True)
    )
    student_log_probs.retain_grad()
    teacher_log_probs = torch.log(torch.tensor([[[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]]]))
    confidence = torch.tensor([[0.25, 0.75, 9.0]], requires_grad=True)
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=torch.zeros((1, 3)),
        teacher_log_probs=torch.zeros((1, 3)),
        response_mask=response_mask,
        self_distillation_config=_config(),
        student_all_log_probs=student_log_probs,
        teacher_all_log_probs=teacher_log_probs,
        distillation_token_weights=confidence,
        distillation_token_weight_mode="teacher_entropy",
        batch_num_tokens=2,
    )

    per_token_jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(-1)
    expected_sum = per_token_jsd[0, 0] * 0.25 + per_token_jsd[0, 1] * 0.75
    assert torch.allclose(loss, expected_sum / 2)
    assert metrics["_self_distillation/entropy_weighted_loss_sum"] == pytest.approx(expected_sum.item())

    loss.backward()
    assert student_log_probs.grad is not None
    assert confidence.grad is None


def test_generic_token_weights_require_a_valid_mode_and_finite_values():
    inputs = {
        "student_log_probs": torch.zeros((1, 1)),
        "teacher_log_probs": torch.zeros((1, 1)),
        "response_mask": torch.ones((1, 1)),
        "self_distillation_config": _config(),
        "student_all_log_probs": torch.log(torch.tensor([[[0.5, 0.5]]])),
        "teacher_all_log_probs": torch.log(torch.tensor([[[0.5, 0.5]]])),
    }
    with pytest.raises(ValueError, match="distillation_token_weight_mode"):
        compute_self_distillation_loss(**inputs, distillation_token_weights=torch.ones((1, 1)))
    with pytest.raises(ValueError, match="finite"):
        compute_self_distillation_loss(
            **inputs,
            distillation_token_weights=torch.tensor([[float("nan")]]),
            distillation_token_weight_mode="teacher_entropy",
        )


def test_beta_zero_unit_weights_match_unweighted_loss_and_gradient():
    teacher_log_probs = torch.log(torch.tensor([[[0.7, 0.3], [0.1, 0.9]]]))

    baseline_logits = torch.tensor([[[0.2, -0.1], [0.4, 0.3]]], requires_grad=True)
    baseline_log_probs = torch.log_softmax(baseline_logits, dim=-1)
    baseline_loss, _ = compute_self_distillation_loss(
        student_log_probs=torch.zeros((1, 2)),
        teacher_log_probs=torch.zeros((1, 2)),
        response_mask=torch.ones((1, 2)),
        self_distillation_config=_config(),
        student_all_log_probs=baseline_log_probs,
        teacher_all_log_probs=teacher_log_probs,
    )
    baseline_loss.backward()

    weighted_logits = baseline_logits.detach().clone().requires_grad_(True)
    weighted_log_probs = torch.log_softmax(weighted_logits, dim=-1)
    weighted_loss, _ = compute_self_distillation_loss(
        student_log_probs=torch.zeros((1, 2)),
        teacher_log_probs=torch.zeros((1, 2)),
        response_mask=torch.ones((1, 2)),
        self_distillation_config=_config(),
        student_all_log_probs=weighted_log_probs,
        teacher_all_log_probs=teacher_log_probs,
        distillation_token_weights=torch.ones((1, 2)),
        distillation_token_weight_mode="teacher_entropy",
    )
    weighted_loss.backward()

    assert torch.equal(weighted_loss, baseline_loss)
    assert torch.equal(weighted_logits.grad, baseline_logits.grad)


def test_loss_path_records_raw_jsd_against_entropy_weights():
    student_log_probs = torch.log(torch.tensor([[[0.8, 0.2], [0.4, 0.6]]], requires_grad=True))
    teacher_log_probs = torch.log(torch.tensor([[[0.5, 0.5], [0.1, 0.9]]]))
    weights = torch.tensor([[0.5, 1.5]])
    accumulator = DistillationTokenWeightMetricsAccumulator(
        mode="teacher_entropy",
        global_token_count=torch.tensor(2.0, dtype=torch.float64),
        sample_size=8,
        vocab_size=2,
    )

    compute_self_distillation_loss(
        student_log_probs=torch.zeros((1, 2)),
        teacher_log_probs=torch.zeros((1, 2)),
        response_mask=torch.ones((1, 2)),
        self_distillation_config=_config(),
        student_all_log_probs=student_log_probs,
        teacher_all_log_probs=teacher_log_probs,
        distillation_token_weights=weights,
        distillation_token_weight_mode="teacher_entropy",
        distillation_token_weight_source=torch.tensor([[0.1, 0.7]]),
        distillation_token_weight_metrics_accumulator=accumulator,
    )
    metrics = accumulator.finalize()

    assert metrics["self_distillation/token_weight_mean"] == 1.0
    assert "self_distillation/token_weight_jsd_pearson" in metrics
