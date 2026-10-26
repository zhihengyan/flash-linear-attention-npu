# RecurrentGatedDeltaRule 接口索引

本次 A5 UB/缓冲实验沿用既有公开接口。参数和支持范围见
[`../README.md`](../README.md#3-参数说明)，ACLNN 接口见
[`aclnnRecurrentGatedDeltaRule.md`](aclnnRecurrentGatedDeltaRule.md)。

- K/V 维度固定 128；Q/K/V/beta/out 为 BF16，state 为 BF16 或 FP32。
- `actualSeqLengths[0]` 为无效前缀，后续为每序列的长度 `[0,8]`，全部项之和为 T。
  `T=1` 能保证所有有效序列最多一个 token；`T=B` 不能提供这个保证。
- g/gk 至少提供一个；g/gk 的 exp 和同序列多 token state 递推在 FP32 中计算。
- 每个有效 token 都写出其 state 槽位的快照，BF16 快照不代替后继 token 的 FP32 state。
- GVA 映射、accepted 初始槽位、非连续输入/state、索引唯一性及 stream 语义保持既有约定。

CPU 标杆继续使用
[`reference_recurrent_gated_delta_rule.py`](../../../../../../../tests/atk/recurrent_gated_delta_rule/reference_recurrent_gated_delta_rule.py)，
输入值域和 `mixed_tolerance_bm` 不变。
