# Profiling 开发期追溯

设计、原始流程与输入契约分别在 [design.md](design.md)、[timer_flow.md](timer_flow.md)、
[api.md](api.md)。本轮用户授权仅接入 profiling 和静态检查；未完成阶段 5 的设备验收，
因此保留开发期记录。当前结论、独立审核证据和设备状态统一在 [timer_report.md](timer_report.md)。

| 设计项 | 代码与验证入口 | 当前证据 |
| --- | --- | --- |
| 原 Stage 0 数学/UB/调度 | arch35 VF、InitLocalBuffers、host tiling processor | 完整 VF/UB init 逐字同 main，剥离打点后的 class 语句同 main |
| timer 独立 GM、槽位与核所有权 | AscendTimerV2.hpp、AscendTimerV2_device.hpp、recurrent_gated_delta_rule_timer.h | 128B 行隔离、linear AIV、配对/容量 CPU 校验，独立审核通过 |
| 既有同步与诊断 drain | arch35 原 DeQue/PIPE_V/V_MTE2/LocalMemBar；入口总 Tok | 原同步保留、无分阶段新 barrier，仅诊断末尾一次 drain；设备扰动待测 |
| host/私有 adapter 接入 | Profile C 头、L2/L0/def/kernel、stable profile adapter | 参数对拍、39 adapter schema parity、strict coverage 通过；CANN 编译待内网 |
| 原输入/精度标准/输出状态 | 原 perf/accuracy spec、原 executor，profile_performance.py | spec 相同静态检查、同输入两份 state 布局复制；真实 NPU 比较待内网 |
| 原性能调用与额外 task | 原 ATK performance_device / op_summary 汇总 | 采集代码就绪，实际主核/ViewCopy Task Duration 尚无数据 |
| 采集完整性与失败行为 | scripts/test_timer_layout.py、target .o/真实调用次数校验 | 10 项 CPU 检查、负向样本拒绝；不冒充设备验证 |
| 版本、安装和环境 | timer_run.md、artifact_check/source_hash/环境校验 | 操作与检查入口齐备，内网容器/CANN/产物路径由实际环境传入 |

采用 CHUNK_DEPENDENT/V1 开发参考（来源 commit a14961e9dfe8c8264768d67fb447101873da4ecf）
核对递推状态与队列生命周期；本算子没有 Cube 或跨核 workspace，不迁移参考的流水常量。
CPU 标杆、值域与 mixed_tolerance_bm 不变，无需为采数另定数学预期。
后续设备验收、覆盖和归档继续按仓库 05 / ATK README，不将本轮源码检查标记为全量验收完成。
