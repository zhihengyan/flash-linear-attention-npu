# Decode 阶段的 tail-scratch 优化

优化的核心是：最后一个 token 算完后，后面已经没有 token 要用这份临时数据，就不再把它存一遍。

原来的内核会把更新后的 FP32 state 写进 UB（芯片内的临时存储区），供下一个 token 继续递推。
这次针对单 token 的 decode 路径，省掉第二遍计算结束时的这次 UB 写入。attention 输出和要保存到
全局显存的 BF16 state 都直接使用更新后的寄存器结果，仍然正常计算、写回。

前一遍计算写入 UB 的门控后 state 仍要保留，因为第二遍计算还要读取它；省掉的只是最后一次、
已经没有后续使用者的 scratch 写入。按 `128 × 128 × 4` 字节计算，每个有效 work 少写 64 KiB UB，
UB 分配大小和全局显存读写量保持原样。

本分支同时带上了这条专用路径所需的改动：

- host 根据条件选择单 work 模板，kernel 直接处理一次任务，省去通用任务循环和重复的索引读取。
- 在读取门控和 beta 的标量值之前，保留 `V_S` 等待，确保前面的 Vector 操作已经完成。
- 继续检查设备端的有效序列长度和 state 索引；空任务直接返回，state 索引不假定为零。

这条路径只用于 A5（`ascend950`）：`B=T=Nk=Nv=1`、`K=V=vStep=128`、BF16 state、单核、
`stateStride2=128`，提供 `g`，不提供 `gk` 和 `numAcceptedTokens`。其他情况继续走通用路径，
保留下一 token 递推需要的 scratch 写入。

prefill 的 K32 优化来自 `main-k32`，decode 源码来自 `codex/gdn-a5-tail-scratch-elision`。
本次合并仅做本地静态核对，合并后的精度和性能仍需在 A5 上实测。
