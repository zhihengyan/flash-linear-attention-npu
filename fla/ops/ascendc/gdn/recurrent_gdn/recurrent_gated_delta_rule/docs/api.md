# RecurrentGatedDeltaRule 接口与诊断边界

公开算子沿用 [README](../README.md) 与 [ACLNN 文档](aclnnRecurrentGatedDeltaRule.md)，
数学、shape、dtype、state 原地更新和逐 token 快照不变。标杆仍为
[`reference_recurrent_gated_delta_rule.py`](../../../../../../../tests/atk/recurrent_gated_delta_rule/reference_recurrent_gated_delta_rule.py)。

本分支只新增独立诊断入口 `aclnnRecurrentGatedDeltaRuleProfile*` 与私有 dispatcher
`_profile_recurrent_gated_delta_rule`，无公开 Python 导出；常规 ACLNN 与 stable schema 不追加参数。
Profile 增加一个 caller 创建且清零的 int64 一维 contiguous timer，长度至少为
`AscendTimer::TOTAL_BUFFER_SIZE`，地址 128B 对齐，与其它输入输出不能别名，并与输入同设备。
timer 是诊断写入张量。Profile 仅支持 A5，拒绝 A2/A3；普通调用继续覆盖原支持范围。
stable 适配层执行设备/对齐/不别名检查；直接 C 调用者必须自行保证这三个前置条件，
L2 只校验 dtype、shape、长度和 contiguous，不能将 C 接口当作完整的地址所有权检查器。

内部 OpDef 增加 optional timer 输入，普通 L0 传 nullptr；这需要完整重建本分支 wheel/OPP。
本轮不改公开 def 的既有参数顺序、ACLNN 既有符号原型、torch schema 或 CPU 标杆。
不支持诊断 graph capture/autograd/并发共用同一 timer；各 launch 使用独立清零缓冲。
