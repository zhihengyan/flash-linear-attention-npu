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
该模式按官方说明只显示 6 个采样核，不能据此宣称获得所有 32 核的完整时序。

本机已通过 case 1、calls=1 的 dry-run、3 份变更文档的 8 个本地链接（含锚点）、
PipeTimeline 命令参数一致性和 git diff --check；确认 .h/.cpp/.py/.json 与 93f5c70a 无差异。
没有本机 A5 或 msprof，不声称已经完成设备采集；需在上轮 Ascend950DT_9582 环境执行命令。
真实流水待回传后分析，其耗时不与 BasicInfo 混为性能样本。

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
