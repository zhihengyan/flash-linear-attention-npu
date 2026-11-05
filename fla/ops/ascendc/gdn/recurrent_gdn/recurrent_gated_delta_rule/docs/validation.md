# A5 多头 decode 开发记录

基线 f918240b，原分支 codex/gdn-k32-tail-scratch-elision 保持该提交。
原实验分支 codex/gdn-a5-multihead-tail-scratch 从其创建，性能 JSON 原样继承。
当前采集分支 codex/gdn-a5-tail-pmu-profile 从 93f5c70a 创建，原实验分支保持 247efa15。
本分支源码、JSON 和 Python 脚本与 93f5c70a 一致；仅更新文档，撤回设计中尚未采用的多头调度。
本机为 Windows，无 torch/ATK/CANN/NPU 或 C++ 编译器；此前交付编码和离线检查，本轮补充采集文档。
采用 CHUNK_DEPENDENT/V1，来源 a14961e9dfe8c8264768d67fb447101873da4ecf；
仅参考递推 scratch 生命周期、任务与事件闭环，不移植其 Cube/GM workspace 架构。

| 改动 | 静态验证 | 设备验证 |
|---|---|---|
| case 1 JSON | 对照原分支 Git blob | 运行 case 1；原文件保持不变 |
| msprof op 脚本 | case 1 加载、参数校验、CSV 过滤/计数/统计、模拟调用边界 | A5 真实采集，核对目标任务/样本数 |
| 末 token scratch | true/false VF 展开、四个 gate 分支、普通/末 token 分派 | BF16/FP32、g/gk/both、多 token、多 V 分片 |
| 多头单 token 调度 | 已撤回，当前源码没有该分支 | 本轮不做该实验 |
| 原路径 | 原单 work、UB、Copy helpers、公共 ABI/A2/A3 与基线比较 | case 0、原 200 精度场景及 MSS/确定性 |

性能 case 1：20 次独立 state 调用，msprof warm-up=20、launch-count=20、BasicInfo；
只统计 RGDR Task Duration(us)，与相同条件下基线交错比较。原来未提供该 shape 的设备基线或目标幅度。
理论末次 Store 少 2 MiB UB/调用，不据此换算延迟或吞吐改善。

开发顺序：先提交脚本/基线准备，再提交末 token 改动，最后提交多头调度。
保留中间提交以便在设备上分别归因两项优化。静态检查通过不代表设备精度/性能通过。
上述为原实验顺序；当前采集分支停留在末 token 改动。

已完成：测评脚本 12 项离线检查通过，包括当前 runtime 返回多个 OPP vendor 的适配、
首次目标调用前全部 state 克隆及 CSV 计数/单位/过滤负向检查。
末 token 改动的 4 项源码检查通过：VF 数学保持、四个 gate 分支参数、仅末 token 分派变化、
UB/搬运与公开接口/A2A3 源码保持。以上为本机离线结果；随后用户回传的设备性能见下一节。

## 2026-11-05：回到末 token 省写基线并准备 PMU 采集

下表为用户回传的 case 1 均值，使用相同脚本和命令参数；不代表独立重复轮次或完整设备验收。

| 版本 | 平均 Task Duration(us) |
|---|---:|
| main | 6.60 |
| 93f5c70a，仅末 token 省写 | 6.55 |
| 9a515a70，再加多头直接分发 | 6.71 |
| 94d3ab97，移除设备重复取余 | 6.57 |

用户确认回到 93f5c70a。保留原实验分支供追溯，不重置其历史。
末 token updated scratch 没有后续读取者；第一遍 scratch、GM 输出与原同步仍保留。
该语义独立性不能保证所有编译产物和输入下延迟绝不回退；现有数据按基本持平处理。
32 个 head 累计少写 2 MiB UB；若每核一个 head，则每核少写 64 KiB，并非每核少写 2 MiB。

下一步按 [采集说明](msprof-metrics.md) 分别采集 BasicInfo 和 PipeUtilization,MemoryUB。
原脚本只汇总 OpBasicInfo；PMU CSV 单独保留，并按目标任务、block_id/sub_block_id 分析。
本机不能执行 A5 采集，尚无 PMU、样本分布、独立重复轮次或设备精度结论；不归档阶段 5 完成。
本轮 case 1 `--dry-run`、5 份变更文档的 11 个本地链接和代码块配对检查通过，
`git diff --check` 通过；确认 `.h/.cpp/.py/.json` 与 93f5c70a 无差异。

## 2026-11-05：PipeTimeline 采集准备

当前分支补齐 [PipeTimeline 步骤](msprof-metrics.md#6-pipetimeline采集并查看流水)。
原脚本支持 calls=1，工具直接采样流水，本轮仅更新文档，未增加内核标记、计时张量或 Python 参数。
内核、性能 JSON 和测评脚本继续与 93f5c70a 一致；若沿用上一轮相同环境和安装对象，无需重新编译。
固定 case 1、device 0、calls=launch-count=1、warm-up=20，输出新目录并保留完整原始结果。
该模式只覆盖部分采样核，不能据此宣称获得所有 32 核的完整时序；实际轨道数见后续回传记录。

本机已通过 case 1、calls=1 的 dry-run、3 份变更文档的 8 个本地链接（含锚点）、
PipeTimeline 命令参数一致性和 git diff --check；确认 .h/.cpp/.py/.json 与 93f5c70a 无差异。
没有本机 A5 或 msprof，不声称已经完成设备采集；需在上轮 Ascend950DT_9582 环境执行命令。
真实流水待回传后分析，其耗时不与 BasicInfo 混为性能样本。

## 2026-11-05：Ascend950DT_9582 的 PipeTimeline 回传

输入为用户提供的 D:\result\msprof_op\rgdr_timeline_JcLct7oc，仅与本环境的前轮 PMU 核对身份，
不与此前其他设备的均值比较。application_complete，case 1、calls=completed_calls=1、Block Dim=32。
元数据记录源码 0a1417d2；case、executor、运行时和安装对象哈希与前轮一致，
dump 二进制 SHA256 为 8c9dcb6cf4ab551713590de366e37f14047b6fda2aa25f4cf1ff7c3a3d343293。

trace 共 65 个完整区间，SCALAR/VECTOR/MTE2/MTE3 分别为 40/12/9/4。
实际有 core0.veccore0、core0.veccore1、core3.veccore0、core3.veccore1 四条 AIV 轨道，
据此修正采集说明：不能按固定 6 条验收，也不能按逻辑 block_id 直接解释这些硬件轨道名。

| 可见区间 | 四条轨道的范围/us |
|---|---:|
| 轨道最早事件至首个 VECTOR | 2.861–3.281 |
| 前两段 MTE2 之间，主要可见 Scalar | 1.270–1.551 |
| 最长 VECTOR 活动区间 | 1.402–1.544 |
| 末段 MTE3 至最后事件 | 0.747–1.078 |

这些是采样区间，尚无 PC/源码阶段标签；不能直接称为某函数、纯等待、VF 本体或可回收延迟。
OpBasicInfo 的 6.418674 us 仅为本次诊断调用，不加入性能基线统计。
下一候选优先检查单 token 下 cuSeqlens/state index 的元数据复用和本核 head 遍历范围。
源码存在重复读取和归属扫描，不代表编译器一定保留等量机器指令；需独立实验或编译结果确认。
本轮完成原始 JSON/CSV/二进制身份、65 个区间和逐轨重叠检查，不修改内核、JSON 或测评 Python。
此处不补记设备精度通过或正式性能优化验收。

## 2026-11-05：元数据复用实验计划

用户确认继续在当前分支实验。对照基线为 a6d33dbb（内核同 93f5c70a），
固定上一轮 Ascend950DT_9582、case 1、BasicInfo calls/warm-up/launch-count=20/20/20。
本环境初始均值 6.472764 us、中位数 6.502314 us、CV 4.2365%；其他设备数据不参与对照。
候选只改 arch35 的 Scalar 元数据复用，不叠加 head 调度、VF、缓冲或同步变化。

| 设计差异 | 边界和验证 |
|---|---|
| ComputeAvgload 返回最后读到的序列长度，仅 B=1 复用 | B=0、B=1、多个序列、空序列在首/中/尾；多序列原读取与负载分配保持 |
| 单 token 输出 state index 复用初始索引 | accepted 缺失/合法值 1、非法值提前退出；非零 prefix/index、padding stride、多 V 分片 |
| 多 token 保留逐 token 索引 | accepted 选择初始 slot、不同输出 slot、稀疏/逆序 index、多 token 原回归 |
| VF/队列/同步与 ABI 保持 | 对照 a6d33dbb 的函数和文件；原 200 条精度及 case 1 设备验收待执行 |

同一 kernel 的只读元数据不由输出写回修改；缓存只在当前 Scalar 调用/序列内存活。
理论减少 case 1 每核一次序列长度读取和每个 V 分片一次输出索引读取，tensor GM 字节数不减少。
机器码可能已合并旧读取，新增条件和寄存器生存期也可能抵消收益，不预设一定加速。
保持 CHUNK_DEPENDENT/V1 参考；本轮只核对递推状态和地址生命周期，不引入参考算子的 Cube/workspace 结构。

本机离线检查：19 项中 18 项通过，1 项因缺少 C++17 编译器跳过。
新增的可选 C++ harness 抽取生产控制函数，对照 a6d33dbb 的地址、任务、gate 参数和访问次数；
本机未执行该 harness，不能记为 Scalar 控制执行验证通过。首次运行曾遇临时目录权限错误，
改用可写的任务临时目录后测试通过，未修改测评程序来规避该环境问题。
VF、原 single-work、分核、全部 Copy helpers、ABI、A2/A3、prefill K32、三份 JSON 和 msprof 脚本保持。

188 个 completed issue、21 条规则映射检查通过。四层静态脚本返回 42 条既有 DFX-MESSAGE error，
位置在未改动的 host/API 文件（例如 aclnn 实现 118 行）；另有 1 条未改动算子 README 缺少“支持范围”概念的提示。
不是零告警检视，本轮没有改动公开参数校验和报错格式。
#362：缓存不假设零 prefix/index 或连续 stride；#365/#429：ATK runner、导入和 OPP 初始化保持；
#640：浮点 VF、Exp/Cast 与顺序保持，设备精度/非有限值结果待执行；#641：host 及原设备非法元数据 guard 保持。
本轮未取得 CANN 编译、NPU 精度、MSS/确定性或性能结果，候选收益尚未验证。
同环境编译安装、case 1 与原精度回归、BasicInfo 交错对照见 [实验步骤](metadata-reuse-experiment.md)。

## 2026-11-05：元数据复用第 1 轮性能回传

候选目录 metadata_reuse_r1_CQCd3Lcc，application 记录源码 c031ed24、工作区干净、20 次调用完成。
设备为 Ascend950DT_9582，case 1、Block Dim=32、全部报告频率 1650。
与本环境旧 BasicInfo 目录 rgdr_basic_RkRSczLb 的 case、executor、框架/设备/运行时字段一致。
20 条目标 CSV 已重算，与用户 summary 完全一致。
20 份 dump 二进制均为 a243129a644762ce13db954a35336815d409b911083e94805a41fa4316b7c45b，
与本轮安装对象记录一致，并非沿用基线的 8c9dcb6c… 对象。

| 单轮指标 | 本环境历史基线 | c031ed24 候选 |
|---|---:|---:|
| count | 20 | 20 |
| mean/us | 6.472764 | 6.583398 |
| median/us | 6.502314 | 6.659074 |
| std/us | 0.274220 | 0.274915 |
| CV | 4.2365% | 4.1759% |

均值增加 0.110634 us（1.71%），中位数增加 0.156761 us（2.41%），本轮未显示收益。
用户确认没有紧邻采集的 a6d33dbb 基线；不能将一轮非交错对照认定为稳定回退或确定原因，
也不能因变化小于 CV 就认定只是随机波动。此前其他设备数据及 PMU/Timeline 时延不参与对照。
ELF 可执行 .text 由 39104 B 增至 39360 B，但代码大小不等于动态指令数或时延原因。
本目录没有候选 PMU/流水或独立精度结果，不据此判定 Scalar 瓶颈变化或精度通过/失败。
保留候选提交供归因，当前不记录为已证实收益的优化；本轮只更新结果，不叠加新 kernel 改动。
下一对照优先重采同环境基线，按 [实验步骤](metadata-reuse-experiment.md) 交错测量独立轮次。

## 内网执行

先为相应提交编译并安装单算子 run 包，并使用新 Python 进程：

```bash
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
./build_out/fla_npu_linux-x86_64.run --install
python scripts/verify_libcust_opapi_md5.py --run-package build_out/fla_npu_linux-x86_64.run
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=mssanitizer
```

run 包路径按实际架构替换。原 MSS 文件尚不能代表全部新运行时分支，补齐代表输入与设备选择证据后
才能记为正式验收。case 1 的精度须单独验证，同一输入调用 CPU 标杆与未被 profiler 重放的 NPU 各一次。
全部有效结果通过后再做 README 的 msprof op 采集；硬件故障、超时或精度失败按仓库精度定位路由处理。
NPU 验证完成前保留本记录，不归档“阶段 5 已完成”。
