# RecurrentGatedDeltaRule 接口与实验边界

本轮沿用 [算子 README](../README.md) 和 [ACLNN 接口](aclnnRecurrentGatedDeltaRule.md) 的全部契约。
CPU 标杆仍为 `tests/atk/recurrent_gated_delta_rule/reference_recurrent_gated_delta_rule.py`，
精度规则仍为 ATK `mixed_tolerance_bm`，不修改输入值域、输出顺序或 state 的原地更新语义。

实验只增加 A5 的内部单 work 模板。公开 ACLNN/torch 签名、OpDef、tiling struct 的字段及布局不变。
只有 A5、BF16 state、B=T=Nk=Nv=1、K=V=vStep=128、blockDim=1、stateStride2=128、
g-only、无 accepted-token 输入时选择它。
这些是 host 可验证的条件，不代表 host 已知设备上的序列长度、前缀或 state index 值。

专用路径仍读取设备侧 prefix/length；有效 token 仅在 prefix=0、length=1 时执行，
prefix=1、length=0 的合法空工作保持无 state 写入。state index 仍从输入读取，不能假定为零；
state block stride 和 tensor view offset 继续沿原调用链处理。

其它 dtype、gate、accepted-token、shape、分片和 SoC 使用原通用路径。没有新增 profiling 输入或符号。
