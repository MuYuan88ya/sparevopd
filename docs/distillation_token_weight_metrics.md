# 蒸馏 Token 权重指标

正负教师模式与正教师熵模式都会为每个有效 SDPO 响应 Token 生成一个非负权重。本文件说明训练日志中与这些权重有关的指标、计算范围和诊断方式。

所有指标只统计：

```text
response_mask × self_distillation_mask == 1
```

的 Token。Prompt、padding、视觉占位 Token 和未路由到 SDPO 的样本不参与统计。统计范围是一次 optimizer step 对应的全局 mini-batch，包含该 mini-batch 的全部 micro-batch 和数据并行 rank。

训练 logger 会使用 `global_steps` 写入这些标量，因此可以直接在 WandB 等后端查看随训练步变化的曲线。实现不保存跨训练步直方图或逐 Token 历史。

## 统一权重指标

以下指标同时适用于 `contrastive_visual_advantage` 和 `teacher_entropy_reweighting`。

| 指标 | 计算与含义 |
| --- | --- |
| `self_distillation/token_weight_mean` | 全局有效 Token 的平均权重。熵模式经过全局归一化后应接近 `1`。 |
| `self_distillation/token_weight_std` | 权重标准差。持续增大通常表示监督逐渐集中。 |
| `self_distillation/token_weight_min/max` | 全局最小和最大权重。 |
| `self_distillation/token_weight_p10/p25/p50/p75/p90/p99` | 权重分位数，用于观察中位数下降、长尾增大等权重塌缩现象。 |
| `self_distillation/token_weight_zero_fraction` | 权重严格为零的全局 Token 比例。该指标精确统计。 |
| `self_distillation/token_weight_below_0_1_fraction` | 权重小于 `0.1` 的 Token 比例。 |
| `self_distillation/token_weight_below_0_5_fraction` | 权重小于 `0.5` 的 Token 比例。 |
| `self_distillation/token_weight_above_2_fraction` | 权重大于 `2` 的 Token 比例。 |
| `self_distillation/token_weight_above_5_fraction` | 权重大于 `5` 的 Token 比例。 |
| `self_distillation/token_weight_effective_token_fraction` | 权重有效样本量除以 Token 总数，见下式。越接近 `1` 越均匀，越接近 `0` 越集中。 |
| `self_distillation/token_weight_top_10pct_mass_fraction` | 权重最高的 10% Token 占总权重的比例。 |
| `self_distillation/token_weight_top_1pct_mass_fraction` | 权重最高的 1% Token 占总权重的比例。 |

有效 Token 比例定义为：

\[
\mathrm{effective\_fraction}
=
\frac{(\sum_t w_t)^2}{N\sum_t w_t^2}
\]

例如该值为 `0.25`，表示当前权重集中程度相当于只使用约 25% 的等权 Token。

分位数、固定阈值比例和 top-k 权重质量使用每个全局 mini-batch 最多 2048 个确定性均匀样本估计。均值、标准差、min/max、零权重比例和有效 Token 比例通过跨 rank 标量归约精确计算。采样使用独立的确定性优先级，不消耗训练随机数状态。

## 权重与蒸馏 Loss 的关系

| 指标 | 计算与含义 |
| --- | --- |
| `self_distillation/token_weight_jsd_pearson` | 权重与加权前逐 Token JSD 的 Pearson 相关系数。 |
| `self_distillation/weighted_to_raw_jsd_ratio` | `sum(weight × raw_jsd) / sum(raw_jsd)`。熵模式使用全局归一化后的权重。 |
| `self_distillation/top_weight_decile_loss_fraction` | 权重最高的 10% Token 占原始 JSD 总量的比例。 |

这些指标只分析 Token 权重本身造成的影响，使用 IS clip 和 rollout correction 之前的原始逐 Token JSD。

- `token_weight_jsd_pearson > 0`：高权重更倾向于 Student 与 Teacher 分歧较大的 Token。
- `token_weight_jsd_pearson < 0`：高权重更倾向于已经较容易、原始 JSD 较小的 Token。
- 有效 Token 比例持续下降，同时 `weighted_to_raw_jsd_ratio` 下降：权重可能正在集中到低价值 Token。
- top 10% 权重质量很高，但 top decile loss 比例很低：少数 Token 获得很大权重，却没有携带相应的蒸馏误差。

## 正负教师指标

正负教师的原始 margin 为：

\[
m_t=\log p_+(y_t)-\log p_-(y_t)
\]

用于 loss 的权重为 `max(0, margin)`，但 margin 指标保留负值，以便观察正教师输给负教师的 Token。

| 指标 | 计算与含义 |
| --- | --- |
| `self_distillation/visual_advantage_margin_mean/std/min/max` | 全局原始正负教师 margin 统计。 |
| `self_distillation/visual_advantage_margin_p10/p25/p50/p75/p90/p99` | 原始 margin 分位数。 |
| `self_distillation/positive_teacher_win_fraction` | `margin > 0` 的精确 Token 比例，即正教师比负教师更支持当前生成 Token 的比例。 |
| `self_distillation/positive_teacher_log_prob_mean` | 正教师对实际生成 Token 的平均 log-prob。 |
| `self_distillation/negative_teacher_log_prob_mean` | 负教师对实际生成 Token 的平均 log-prob。 |
| `self_distillation/positive_negative_log_prob_gap_mean` | 上述两个均值之差，等于平均原始 margin。 |

如果 `positive_teacher_win_fraction` 持续下降，说明正教师视觉条件相对负教师的优势正在减弱。如果非零权重比例很低但 `p99` 很高，说明只有极少数响应 Token 能明显区分正负视觉条件。

## 正教师熵指标

Teacher 熵使用当前 Teacher forward 温度下的完整词表分布精确计算：

\[
H_t=-\sum_v p_t(v)\log p_t(v)
\]

| 指标 | 计算与含义 |
| --- | --- |
| `self_distillation/teacher_entropy_mean/min/max` | 已有的全局 Teacher 熵统计。 |
| `self_distillation/teacher_entropy_std` | 全局 Teacher 熵标准差。 |
| `self_distillation/teacher_entropy_p10/p25/p50/p75/p90/p99` | Teacher 熵分位数。 |
| `self_distillation/teacher_normalized_entropy_mean` | `entropy / log(vocab_size)` 的全局均值，范围理论上为 `[0, 1]`。 |
| `self_distillation/teacher_low_entropy_fraction` | 归一化熵不超过 `0.25` 的 Token 比例。 |
| `self_distillation/teacher_medium_entropy_fraction` | 归一化熵位于 `(0.25, 0.60]` 的 Token 比例。 |
| `self_distillation/teacher_high_entropy_fraction` | 归一化熵大于 `0.60` 的 Token 比例。 |

熵均值和中位数持续下降通常表示 Teacher 越来越确定；熵突然升高可能对应 Teacher EMA 更新、数据分布变化或训练不稳定。应将熵曲线与 `token_weight_effective_token_fraction` 一起观察：Teacher 变得更确定并不必然意味着权重更均匀。

## 推荐监控组合

判断权重是否塌缩：

```text
token_weight_p50
token_weight_p90
token_weight_p99
token_weight_effective_token_fraction
token_weight_top_10pct_mass_fraction
```

判断权重是否选择了有价值的监督：

```text
token_weight_jsd_pearson
weighted_to_raw_jsd_ratio
top_weight_decile_loss_fraction
raw_jsd_token_mean
```

判断正负教师是否仍有区分度：

```text
positive_teacher_win_fraction
visual_advantage_margin_p50
visual_advantage_margin_p90
positive_negative_log_prob_gap_mean
```

判断熵控制是否过度集中：

```text
teacher_normalized_entropy_mean
teacher_entropy_p50
teacher_entropy_p90
token_weight_effective_token_fraction
token_weight_above_2_fraction
```
