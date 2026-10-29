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

已保留的尾部 scratch 实验沿用上述选择条件，仅省去专用 VF 内无后续消费者的 updated FP32 UB 写回。
公开 stateRef 的 GM 写回仍执行；通用路径继续保存逐 token 递推所需的 FP32 scratch。

本轮仅在已有 SINGLE_WORK=1 内融合初始 BF16 state 的输入转换。原 state 输入队列持有到
主 VF 消费后释放；general 继续原整体 Cast。没有新增公开参数、key 或接口；g/beta 的
标量读取、V_S、两遍计算和输出规则保持 fec1e18b 的行为。
