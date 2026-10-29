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

本轮仅替换已有 SINGLE_WORK=1 的输入启动路线：小输入复用原 qInQueue 的一个 slot 打包，
state 保留独立队列并提前发射搬入，小输入由一个 VF 准备到原 FP32 区域。
公开参数、ABI、host/key 和输入契约均不变，主递推、state Cast、Scalar g/beta 与 V_S 保持基线。
