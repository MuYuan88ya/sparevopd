from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import compute_self_distillation_loss


def make_config():
    return SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=None,
        distillation_add_tail=False,
        alpha=0.0,
        is_clip=None,
    )


def make_loss_inputs():
    student_all_log_probs = torch.log(
        torch.tensor([[[0.8, 0.2], [0.6, 0.4], [0.5, 0.5]]], requires_grad=True)
    )
    student_all_log_probs.retain_grad()
    teacher_all_log_probs = torch.log(
        torch.tensor([[[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]]])
    )
    return {
        "student_log_probs": torch.zeros((1, 3)),
        "teacher_log_probs": torch.zeros((1, 3)),
        "response_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "self_distillation_config": make_config(),
        "student_all_log_probs": student_all_log_probs,
        "teacher_all_log_probs": teacher_all_log_probs,
    }


def test_visual_advantage_weights_match_opdv_tokenwise_objective():
    inputs = make_loss_inputs()
    visual_advantage_weights = torch.tensor([[0.0, 2.0, 5.0]], requires_grad=True)
    loss, metrics = compute_self_distillation_loss(
        **inputs,
        distillation_token_weights=visual_advantage_weights,
        distillation_token_weight_mode="visual_advantage",
    )

    raw_per_token_loss = F.kl_div(
        inputs["student_all_log_probs"],
        inputs["teacher_all_log_probs"],
        reduction="none",
        log_target=True,
    ).sum(-1)
    expected = (raw_per_token_loss[0, 0] * 0.0 + raw_per_token_loss[0, 1] * 2.0) / 2.0
    assert torch.allclose(loss, expected)
    assert metrics["self_distillation/visual_advantage_token_mean"] == pytest.approx(1.0)
    assert metrics["self_distillation/visual_advantage_nonzero_fraction"] == pytest.approx(0.5)
    assert metrics["self_distillation/visual_advantage_activated_weight_mean"] == pytest.approx(2.0)

    loss.backward()
    assert inputs["student_all_log_probs"].grad is not None
    assert visual_advantage_weights.grad is None


def test_visual_advantage_zero_margin_removes_token_loss_without_renormalizing():
    inputs = make_loss_inputs()
    loss, _ = compute_self_distillation_loss(
        **inputs,
        distillation_token_weights=torch.zeros((1, 3)),
        distillation_token_weight_mode="visual_advantage",
    )
    assert loss.item() == 0.0


def test_visual_advantage_rejects_invalid_shape_and_negative_values():
    inputs = make_loss_inputs()
    with pytest.raises(ValueError, match="must match"):
        compute_self_distillation_loss(
            **inputs,
            distillation_token_weights=torch.ones((1, 2)),
            distillation_token_weight_mode="visual_advantage",
        )
    with pytest.raises(ValueError, match="non-negative"):
        compute_self_distillation_loss(
            **inputs,
            distillation_token_weights=torch.tensor([[1.0, -1.0, 0.0]]),
            distillation_token_weight_mode="visual_advantage",
        )
