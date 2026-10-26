# RecurrentGatedDeltaRule 接口索引

本次 A5 VF 优化沿用既有接口，不增加或改变公开参数、ABI、dtype、layout、shape、
状态槽位或错误语义。完整参数表及调用约束见
[`../README.md`](../README.md#3-参数说明)，ACLNN 接口见
[`aclnnRecurrentGatedDeltaRule.md`](aclnnRecurrentGatedDeltaRule.md)。

与本次内部实现直接相关的既有约定：

- `Dk=Dv=128`；Q/K/V/beta/out 为 BF16，state 为 BF16 或 FP32。
- `g`、`gk` 至少存在一个，分别在 FP32 中计算指数；同时存在时先施加 g，再施加 gk。
- `actualSeqLengths[0]` 为无效前缀长度，后续各项为 `[0,8]` 的序列长度，
  并非累积位置；不得用 `T==B` 推断所有序列长度为 1。
- 每个 value head 使用 `hk=hv/(Nv/Nk)` 的 Q/K。
- 初始 state 槽位由 `ssmStateIndices[seq0+accepted-1]` 选择；未提供 accepted 时取 1。
  每个有效 token 都向其 `ssmStateIndices[token]` 槽位写回 state。
- 同一序列的多 token 递推保留 FP32 中间 state；BF16 写回快照不作为下一 token 的输入。
- 无效前缀和零长度序列沿用既有处理；state 和输入非连续视图的支持条件沿用原接口文档。

CPU 预期继续使用
[`reference_recurrent_gated_delta_rule.py`](../../../../../../../tests/atk/recurrent_gated_delta_rule/reference_recurrent_gated_delta_rule.py)，
本轮不修改标杆、输入生成器或 `mixed_tolerance_bm`。
