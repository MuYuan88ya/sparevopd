import pytest

from verl.workers.config.actor import ActorConfig, PolicyLossConfig, SelfDistillationConfig


def make_sparse_config(**overrides):
    values = {
        "teacher_always_on": True,
        "teacher_visual_token_mode": "same_image_bbox_mask",
        "teacher_visual_token_selection_strategy": "fixed_stratified",
        "teacher_visual_token_selection_kwargs": {"background_keep_ratio": 0.1, "seed": 0},
    }
    values.update(overrides)
    return SelfDistillationConfig(**values)


def test_contrastive_sparse_teacher_defaults_to_disabled():
    config = make_sparse_config(
        teacher_visual_token_selection_kwargs={"background_keep_ratio": 1.0, "seed": 0}
    )
    assert config.contrastive_visual_advantage is False
    assert config.teacher_entropy_reweighting is False
    assert config.teacher_entropy_beta == 1.0


def test_contrastive_sparse_teacher_accepts_supported_configuration():
    config = make_sparse_config(contrastive_visual_advantage=True)
    assert config.contrastive_negative_mode == "complement-random-equal"
    assert config.contrastive_negative_seed == 0


def test_contrastive_sparse_teacher_rejects_background_ratio_above_thirty_percent():
    with pytest.raises(ValueError, match="background_keep_ratio <= 0.30"):
        make_sparse_config(
            contrastive_visual_advantage=True,
            teacher_visual_token_selection_kwargs={"background_keep_ratio": 0.31, "seed": 0},
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"teacher_visual_token_mode": "crop"}, "same_image_bbox_mask"),
        ({"teacher_visual_token_selection_strategy": "custom"}, "fixed_stratified"),
        ({"contrastive_negative_mode": "no-image"}, "complement-random-equal"),
    ],
)
def test_contrastive_sparse_teacher_rejects_unsupported_modes(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_sparse_config(contrastive_visual_advantage=True, **overrides)


def test_entropy_reweighting_accepts_supported_configuration_and_beta_zero():
    config = make_sparse_config(teacher_entropy_reweighting=True, teacher_entropy_beta=0.0)
    assert config.teacher_entropy_reweighting is True
    assert config.teacher_entropy_beta == 0.0


@pytest.mark.parametrize("beta", [-0.1, float("inf"), float("nan")])
def test_entropy_reweighting_rejects_invalid_beta(beta):
    with pytest.raises(ValueError, match="finite and non-negative"):
        make_sparse_config(teacher_entropy_reweighting=True, teacher_entropy_beta=beta)


def test_entropy_reweighting_is_mutually_exclusive_with_contrastive_teacher():
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_sparse_config(
            teacher_entropy_reweighting=True,
            contrastive_visual_advantage=True,
        )


def test_entropy_reweighting_requires_same_image_sparse_teacher():
    with pytest.raises(ValueError, match="same_image_bbox_mask"):
        SelfDistillationConfig(
            teacher_always_on=True,
            teacher_image_key="teacher_images",
            teacher_visual_token_mode="crop",
            teacher_entropy_reweighting=True,
        )


def _actor_config(**overrides):
    values = {
        "strategy": "fsdp",
        "rollout_n": 1,
        "ppo_micro_batch_size_per_gpu": 1,
        "policy_loss": PolicyLossConfig(loss_mode="vopd"),
        "self_distillation": make_sparse_config(teacher_entropy_reweighting=True),
    }
    values.update(overrides)
    return ActorConfig(**values)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"policy_loss": PolicyLossConfig(loss_mode="vanilla")}, "loss_mode='vopd'"),
        ({"loss_agg_mode": "seq-mean-token-mean"}, "loss_agg_mode='token-mean'"),
        ({"entropy_coeff": 0.01}, "entropy_coeff=0"),
        ({"use_kl_loss": True}, "use_kl_loss=False"),
    ],
)
def test_entropy_reweighting_rejects_mixed_gradient_actor_modes(overrides, match):
    with pytest.raises(ValueError, match=match):
        _actor_config(**overrides)
