# RecurrentGatedDeltaRule 接口与本轮边界

公开签名、输入输出和异常约束沿用当前 [算子说明](../README.md) 与
[ACLNN 说明](aclnnRecurrentGatedDeltaRule.md)。本轮不改变接口、tiling 数据布局、SoC 支持或数学语义。
query/key/value/beta/out 为 BF16，state 支持 BF16/FP32，K=V=128；Hv 为 Hk 的整数倍。
T 是无效前缀与所有序列有效长度之和，每条有效序列长度在 [0,8] 内。
state 原地更新，输出仍为 attention；g/gk 至少提供一个，accepted-token 和非连续视图保持原约束。

本轮用户指定性能输入为 `_perf.json` case 1：B=T=1、Hk=16、Hv=32、K=V=128、BF16 state、
g-only、无 accepted-token 输入、连续布局、seed=20260826。JSON 从基线 f918240b 原样继承。

A5 的额外调度分支只处理 BF16 state、B=T=1、Hv>1、K=V=vStep=128、stateStride2=128、
g-only、无 accepted-token 输入。设备端仍要求 prefix=0、length=1，否则直接返回；
不假定 state index 为零，保留 state 的 block/head stride。
原单 work 分支继续保留；其他输入沿用通用调度。

A5 通用调度中，每个序列、每个 V 分片的最后一个 token 可以省去更新后 FP32 scratch 的 UB 写入。
中间 token 仍写入该 scratch，所有 token 的全局 state/out 输出及第一遍门控后 scratch 均保留。
A2/A3 的 kernel 不修改。
