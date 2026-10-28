# RecurrentGatedDeltaRule：仅采数设计

方案设计规则版本：`V2`。本分支 main 基线 `c9e2f1db`，接口见 [api.md](api.md)，
完整原始流程、分核、同步锚点与全部观测点见 [timer_flow.md](timer_flow.md)。

## 1. 目标

为用户确认的 A5 默认 perf case 提供同输入普通/诊断调用对照、每核阶段 cycles、
原 ATK 系统 profile 和可选 Perfetto。只加观测，不改变原运算、归约树、队列深度、UB 预算、
vStep、head 调度、workspace 或输出 ViewCopy。A/B 两轮优化不带入。
用户授权仅编码和静态检查，设备采集在内网完成；不编造新的性能/精度结论。

Host 的公共 workspace 构造 helper 只复用校验、executor 和下发逻辑，不放置 L2 DFX 宏。
普通与 Profile 入口各自持有与函数名一致的 `L2_DFX_PHASE_1/2`，避免 CANN 的入口名静态断言失败。

## 2. Stage 0：Vector，原 recurrent VF 与诊断旁路

数据路径继续是 FP32 gate → state·key reduction → delta → rank-one → state·query reduction
→ state/out cast；K/V=128。每个 sequence/value head 按既有 IsCurrentBlock 分配给单 AIV，
Q/K source head=hv/(Nv/Nk)。默认 case 只有一个 work，blockDim=1，原分片/队列选择由 tiling 决定。
L1/L0/算法 GM workspace=0，系统 workspace 保留原 16 MiB。

原 UB 地址、所有对象份数、消费者与释放时机不变。timer 是独立 int64 GM：
`[0,N_TIMING_COUNTER)` 为按核对齐的事件对，
`[N_TIMING_COUNTER,BASE_BUFFER_SIZE)` 为按核对齐动态次数，
`[BASE_BUFFER_SIZE,HEADER_OFFSET)` 为每核 64 个 int64 metadata，
`[HEADER_OFFSET,TOTAL_BUFFER_SIZE)` 为 16 个 int64 全局头。
核行映射、128B 对齐和每槽索引由 AscendTimerV2.hpp 常量与解析器共同定义；容量 128 核，
每类型最多采样 16 次，metadata 保留未截断计数。计时器不占原 UB，不改变原资源峰值。

每动态类型最多一个在途 Tik；Begin/End 用相同 type/slot，嵌套外窗使用不同类型。
无需跨核 flag；每核只写自己的 timer/metadata 行，头部仅核 0 写，不存在写冲突。
所有起点 NoBarrier；阶段终点 NoBarrier，只在有原 DeQue 锚点时称为 ready 等待。
总窗口末尾增加一次仅诊断路径的 PIPE_ALL drain，之后做每核 DCCI 发布。
总窗口不含发布开销；它也不能代替未打点版本的 msprof Task Duration。

R21：未改 VF、Cast、输入值域或 CPU 标杆，数学风险不增加。R03/R08/R11/R17：原 UB
保持基线；诊断 GM 容量/对齐在 stable/L2/tiling 三侧校验。R01/R20 的任务重映射未引入。
不将多段嵌套 issue window 相加为纯计算总耗时。独立审核与设备状态集中在
[timer_report.md](timer_report.md)，操作入口见 [timer_run.md](timer_run.md)。
