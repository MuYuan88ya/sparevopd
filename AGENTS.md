# 同图稀疏视觉 Token 教师研究说明

## 研究动机

本项目中的稀疏视觉 Token 教师并不是为了减少视觉 Token 开销，也不应被描述为一种推理加速或显存优化方法。

原始 Vision-OPD 使用目标框裁剪图作为教师输入。裁剪图能够突出局部细节，但会切断目标与完整场景之间的联系，使教师缺少全图视野。例如，教师可能看清目标本身，却无法利用目标所在位置、周围物体、场景类别和跨区域关系等上下文信息。

本研究希望在保留目标区域高信息密度的同时，让教师继续感知完整图像。教师与学生输入同一张原图，区别仅在于教师进入语言模型的视觉 Token 经过有偏选择：

- 目标框覆盖的视觉 Token 全部保留，作为教师视觉表征的主体。
- 目标框外保留可配置比例的全局视觉 Token，用于补充场景上下文；默认比例为 10%。
- 全局 Token 应在图像空间中尽量均匀分布，而不是集中在局部区域。
- 文本、视觉边界符和学生生成的回答 Token 不受视觉筛选影响。

因此，教师看到的是“以目标区域为中心、带有稀疏全局上下文的完整图像表征”，而不是一张独立的目标裁剪图。

## 核心假设

目标相关信息主要集中在 bbox 对应的视觉 Token 上，但正确理解目标通常仍需要一定量的全局信息。少量、空间分散的框外 Token 可以提供：

- 目标在完整图像中的位置与尺度。
- 周围物体和目标之间的关系。
- 场景、背景和环境语义。
- 裁剪图中缺失的跨区域参照信息。

预期这种教师会比纯裁剪教师拥有更完整的语义判断能力，同时比全图 dense 教师更突出目标区域。

## 方法边界

- 教师和学生必须使用相同的原始图片、prompt 和图像预处理结果。
- bbox 只用于构造教师视觉 Token keep mask，不作为额外文本提示输入模型。
- 教师保留全部目标 Token，并按固定规则选择一部分框外 Token。
- 视觉 Token 选择由 `teacher_visual_token_selection_strategy` 指定的注册策略完成；策略参数通过 `teacher_visual_token_selection_kwargs` 传入。内置 `fixed_stratified` 的背景保留比例默认是 `0.1`。
- 当前实现仍由 ViT 编码完整图像，筛选发生在合并后的视觉特征进入 LLM 之前。
- 被丢弃的视觉占位符通过 `attention_mask` 和 `unpad_input` 从教师 LLM 序列中移除。
- 教师只评价学生生成的 rollout，执行 `torch.no_grad()` forward，不进行生成或反向传播。
- 学生始终使用完整视觉 Token，训练目标仍是论文中的 on-policy 蒸馏。

## 学生视觉特征选择平台

启用 `same_image_bbox_mask` 后，actor 在每次 `_forward_micro_batch` 调用模型前，根据 bbox、原图尺寸和 `image_grid_thw` 生成目标区域 mask，并创建只存在于当前调用栈的 `VisualTokenSelectionRequest`。请求传给 Qwen3.5 的 `_get_input_embeds`，由模型在完成 ViT 和学生 dense 多模态 embedding 注入后立即调用已注册策略：

- 学生的 dense `image_embeds`。
- 学生完整的 `inputs_embeds` 和紧凑的 `image_mask`。
- 整个 micro-batch 的 `image_grid_thw`、扁平目标 mask、稳定样本编号和空间合并尺寸。

策略只返回最终 keep mask，不返回 selected embedding。学生仍将完整视觉 embedding 注入自己的 LLM；模型输出只携带 detached 的 bool keep mask。actor 按当前 micro-batch 的样本顺序切分并保存 compact selection state，不保存 dense embedding。

随后 actor 用同一份 state 修改教师 `attention_mask` 并写入教师多模态输入。教师 forward 仍在 actor 中先经过现有 `unpad_input`，再由 Qwen3.5 重新完成全图 ViT forward，并用 `image_token_keep_mask` 筛选进入教师 LLM 的视觉特征。这样策略只负责选择，attention-mask/unpadding 和视觉特征对齐仍各自只有一条实现路径。

当前 `fixed_stratified` 策略按照 Morton 空间顺序分层并确定性抽样。学生 embedding 已经进入策略接口并完成形状与 batch 对齐检查，但当前策略不读取 embedding 数值进行排序。因此现阶段不能将方法描述为 embedding-similarity selection；这次改动建立的是后续特征选择实验所需的平台和正确时序。

## 研究表述

实验和论文中应将该方法描述为一种改善教师视觉条件的机制，而不是 Token 剪枝或加速方法。核心对比应回答：

1. 同图稀疏教师是否比裁剪教师更好地利用全局上下文？
2. 目标 Token 加少量全局 Token 是否优于仅目标 Token？
3. 它是否比同图全量 Token 教师更能聚焦目标区域？
4. 全局 Token 比例和空间采样方式如何影响教师质量及蒸馏效果？

计算指标可以用于分析，但不能作为本方法的首要贡献。尤其需要明确：当前方案减少的是教师 LLM 的有效序列长度，ViT 仍然完整处理全图。

## 建议实验

至少比较以下教师条件：

- 原始目标裁剪教师。
- 同图全量视觉 Token 教师。
- 同图仅目标区域 Token 教师。
- 同图目标区域 Token 加 10% 全局 Token 教师。

重点报告教师答案准确率、置信度、教师与学生的 JSD、V*Bench、ZoomBench 和训练稳定性。建议进一步消融全局 Token 保留比例及采样方式，以验证收益来自全图上下文，而不是偶然的 Token 数量变化。
