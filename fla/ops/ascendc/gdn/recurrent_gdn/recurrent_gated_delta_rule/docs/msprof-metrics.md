# A5 decode：PipeUtilization、MemoryUB 采集方法

当前采集分支为 `codex/gdn-a5-tail-pmu-profile`，保留 prefill K32 和末 token scratch 省写。
当前轮以内核同 93f5c70a 的 a6d33dbb 为基线实验 [head 遍历范围](head-range-experiment.md)，
case 1 仍使用原通用计算流程和原 head 归属。每轮按源码提交与安装对象哈希区分版本。
后续多头直接分发实验留在 `codex/gdn-a5-multihead-tail-scratch`，本分支没有该调度。

本轮先回答两个问题：主要时间花在 Scalar、Vector 还是 GM 搬运；省掉的 UB Store 是否影响关键路径。
这两个指标组给出整核汇总，不能直接指出某一行代码或某个同步事件的耗时。

## 1. 固定输入和环境

使用现有 `tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py`，不用修改 Python：

| 参数 | 固定值 |
|---|---|
| 性能用例 | `_perf.json` case 1 |
| B / T / Hk / Hv | 1 / 1 / 16 / 32 |
| K / V | 128 / 128 |
| state / gate | BF16 / g-only |
| 布局、序列、state index | 连续、单个有效 token、index=0 |
| BasicInfo/PMU 的 Python 调用数 / launch-count / warm-up | 20 / 20 / 20 |
| PipeTimeline 的 Python 调用数 / launch-count / warm-up | 1 / 1 / 20 |
| 目标 kernel | `RecurrentGatedDeltaRule` |
| 设备 | Linux + Ascend 950，下面以 device 0 为例 |

先切到本分支，沿用已有方式重新编译、安装，加载 CANN 并激活 ATK/PyTorch 环境。
例如 wheel 构建仍使用：

```bash
FLA_NPU_OPS=recurrent_gated_delta_rule FLA_NPU_SOC=ascend950 python scripts/build_wheel.py
```

执行构建末尾打印的安装命令，再以新 Python 进程采集。仅切 Git 不会替换已安装的算子。
`application.json` 会记录源码提交、case 和安装包/对象哈希，但它不证明安装对象来自该源码；
仍需按既有流程用 `scripts/verify_libcust_opapi_md5.py` 核对本轮构建产物。

以下命令都从仓库根目录执行，示例仓库路径不含空格。先检查工具和输入：

```bash
msprof op --help
SCRIPT="$PWD/tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py"
python "$SCRIPT" --repo-root "$PWD" --case-id 1 --calls 20 --dry-run
```

确认本机 `msprof op` 支持 `PipeUtilization`、`MemoryUB`。参数依据
[msOpProf 26.0 使用说明](https://github.com/Ascend/msopprof/blob/26.0.0/docs/en/user_guide/msopprof_user_guide.md)。
每轮使用新目录，同一设备避免并行运行其他测评。精度独立运行，命令见 [ATK README](../../../../../../../tests/atk/recurrent_gated_delta_rule/README.md)；
profiler 会重放原地更新 state 的 kernel，其输出不作精度结论。

## 2. BasicInfo：延续原来的时延口径

```bash
SCRIPT="$PWD/tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py"
mkdir -p tests/atk/recurrent_gated_delta_rule/runs/msprof_op
TIME_OUT=$(mktemp -d "$PWD/tests/atk/recurrent_gated_delta_rule/runs/msprof_op/rgdr_basic_XXXXXXXX")
msprof op \
  --application="python $SCRIPT --repo-root $PWD --device 0 --case-id 1 --calls 20 --metadata $TIME_OUT/application.json" \
  --kernel-name=RecurrentGatedDeltaRule \
  --launch-count=20 --warm-up=20 --aic-metrics=BasicInfo --output="$TIME_OUT"
python "$SCRIPT" --summarize "$TIME_OUT" --expected-count 20
```

只看目标 RGDR 的 `Task Duration(us)`，要求 20 条，保留 mean/median/std/CV/P90/P99。
比较两个版本时交错采集多轮，每轮独立汇总；同一轮 20 次不能代替独立重复轮次。

## 3. PMU：一起采集两个指标组

父目录同上，仍从仓库根目录执行：

```bash
SCRIPT="$PWD/tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py"
mkdir -p tests/atk/recurrent_gated_delta_rule/runs/msprof_op
PMU_OUT=$(mktemp -d "$PWD/tests/atk/recurrent_gated_delta_rule/runs/msprof_op/rgdr_pmu_XXXXXXXX")
msprof op \
  --application="python $SCRIPT --repo-root $PWD --device 0 --case-id 1 --calls 20 --metadata $PMU_OUT/application.json" \
  --kernel-name=RecurrentGatedDeltaRule \
  --launch-count=20 --warm-up=20 \
  --aic-metrics=PipeUtilization,MemoryUB --output="$PMU_OUT"

find "$PMU_OUT" -type f \( \
  -name 'OpBasicInfo*.csv' -o \
  -name 'PipeUtilization*.csv' -o \
  -name 'MemoryUB*.csv' \)
```

工具会在 `OPPROF_*` 的目标任务目录中生成相应 CSV，保留原目录层级。
用 OpBasicInfo 的 kernel 名、device、任务目录确认目标，别把输入生成的 Cast/Neg 算进去。
不同工具版本的导出目录和附加列可能不同，以实际 CSV 表头为准。

PMU 可能多次重放以取得不同计数器；不要用其时延替代第 2 步 BasicInfo 的结果，
也不要把重放或每核行数当成新的独立调用样本。
现有 `--summarize` 只处理 OpBasicInfo，两个 PMU CSV 按下表直接分析。

## 4. 先看这些字段

下列字段按官方文档的 **Ascend 950** 部分核对，具体含义见
[msOpProf 性能数据说明](https://github.com/Ascend/msopprof/blob/26.0.0/docs/en/user_guide/msopprof_performance_data.md)。
本算子为 AIV-only，优先看 `aiv_*`；Cube 列为空或 N/A 属正常现象。

| 文件 | 字段 | 本轮用途 |
|---|---|---|
| OpBasicInfo | `Task Duration(us)`、`Block Dim`、`Device ID`、`Current Freq` | 总时延、启动的任务块数、设备与频率 |
| PipeUtilization | `block_id`、`sub_block_id`、`aiv_time(us)`、`aiv_total_cycles` | 找最慢核及核间差异 |
| PipeUtilization | `aiv_scalar_time(us)`、`aiv_scalar_ratio` | 判断元数据读取、head 扫描和控制是否值得继续查 |
| PipeUtilization | `aiv_vec_time(us)`、`aiv_vec_ratio` | 判断 Vector 计算、Cast、归约是否占据主要活动时间 |
| PipeUtilization | `aiv_mte2_time(us)`、`aiv_mte2_ratio`、`aiv_mte2_active_bw(GB/s)` | GM→UB 输入搬运 |
| PipeUtilization | `aiv_mte3_time(us)`、`aiv_mte3_ratio`、`aiv_mte3_active_bw(GB/s)` | UB→GM 输出搬运 |
| MemoryUB | `aiv_ub_read_bw_vector(GB/s)`、`aiv_ub_write_bw_vector(GB/s)` | Vector 与 UB 之间的读、写 |
| MemoryUB | `aiv_ub_write_bw_gm(GB/s)`、`aiv_ub_read_bw_gm(GB/s)` | 分别为 GM→UB、UB→GM，注意名称中的方向 |

Pipe 活动可能重叠，比例不能相加当作总时延。Scalar 高不能单独证明某个除法或某次 Wait 就是瓶颈。
Vector 总指标包含多个阶段；若还需区分 VF/SFU，可另开目录加采 `ArithmeticUtilization`。
不要要求 A5 一定输出 A2/A3 文档中的细分 Scalar wait 或 Scalar UB 带宽字段。

MemoryUB 是以整核总周期为分母的带宽，PipeUtilization 的 MTE active bandwidth 使用活动周期；
两类数值分母不同，不直接相减或比较大小判断饱和。
同一任务、同一核、同一 MemoryUB 统计窗口，可粗估：

```text
UB 写入字节数 ≈ aiv_ub_write_bw_vector(GB/s) × aiv_time(us) × 1000
```

不要只比较 GB/s：写入量减少时，如果时间也减少，带宽未必下降。
先逐任务匹配 `block_id/sub_block_id`，看各核 min/median/max 和最慢核，再比较版本。
20 次调用 × 32 核的记录不等于 640 个 kernel 时延样本。

## 5. 怎么据此选下一步

| 观察结果 | 可以继续检查的方向 |
|---|---|
| 慢核 Scalar 时间较长，Vector/MTE 活动较少 | 元数据、地址计算、head 扫描和分派；先确认时间窗口，再选单项实验 |
| Vector 活动突出 | 两遍 VF、归约、Cast、UB 访问；结合 UB 读写量，不能直接归因于最后一个 Store |
| MTE2/MTE3 活动突出 | state 搬运、搬运粒度、计算与搬运的重叠 |
| 少数核明显更慢 | 实际 blockDim、head 归属、每核工作量和 vStep；不把核间差异混成均值 |
| 各 Pipe 活动都少，但总任务仍约 6.5 us | 调度、同步或短任务固定开销；需要时序证据区分 |

若单独研究之前的“末 token 省写”，可用 `7acfe2e1`（脚本已就绪、未推广通用路径省写）
与 `93f5c70a` 做相同条件的 BasicInfo 和 PMU 对照；main 可继续作总体性能参照。
两者 kernel 的差异就是这项省写，更容易解释计数变化。
case 1 理论每 head 少写 `128×128×4=64 KiB` UB，32 head 合计 2 MiB；
GM state 仍每 head 读、写 BF16 各 32 KiB，GM 流量预期不因这项优化下降。
每核只有一个 head 时，其省写量是 64 KiB。vStep 和缓冲配置需从实际 tiling 日志确认，
Block Dim 从 OpBasicInfo 确认，不能仅凭 TilingKey 推断。

若 UB 写入量下降而 BasicInfo 时延持平，说明流量确实可能减少，但尚无证据表明它能缩短关键路径。
也不能把少写 2 MiB 直接换算成若干微秒收益。
当前 head 范围实验使用 a6d33dbb 对照，不用上述历史 Store 实验的版本替代。

汇总指标仍无法区分计算和等待时，按下一节采集 PipeTimeline。

## 6. PipeTimeline：采集并查看流水

### 是否需要改代码或重新编译

现有 Python 脚本已支持 `--calls 1`，能够直接采集，不需要新增 kernel 打点、修改 ABI 或计时张量，
也不需要为 PipeTimeline 添加 `-g`。这是工具采集能力，不表示后续优化版本可以跳过重新构建。
当前 head 范围实验已修改 kernel，须先按第 1 节重新编译、安装并核对本轮产物。
仅在采集同一份已验证安装对象且源码/构建没有变化时，才可直接复用产物。

目标仍是同一环境下的 case 1：B=T=1、Hk=16、Hv=32、BF16 state、g-only。
先采一条代表调用用于查看时序，`--calls` 和 `--launch-count` 都设为 1，工具预热仍为 20。
在上一轮 Ascend950DT_9582 环境继续采集；其他型号的数据单独保存和分析。

### 采集步骤

在 Linux A5 环境加载 CANN、激活上一轮 ATK/PyTorch 环境，在仓库根目录检查：

```bash
msprof op --help
SCRIPT="$PWD/tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py"
python "$SCRIPT" --repo-root "$PWD" --case-id 1 --calls 1 --dry-run
```

确认帮助中支持 `PipeTimeline`，离线配置仍为 case 1 后，执行：

```bash
SCRIPT="$PWD/tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py"
mkdir -p tests/atk/recurrent_gated_delta_rule/runs/msprof_op
OUT=$(mktemp -d "$PWD/tests/atk/recurrent_gated_delta_rule/runs/msprof_op/rgdr_timeline_XXXXXXXX")
msprof op \
  --application="python $SCRIPT --repo-root $PWD --device 0 --case-id 1 --calls 1 --metadata $OUT/application.json" \
  --kernel-name=RecurrentGatedDeltaRule \
  --launch-count=1 --warm-up=20 --aic-metrics=PipeTimeline --output="$OUT"
```

与前面的命令一样，示例假定仓库路径不含空格，每次使用新的输出目录。
检查 msprof 正常结束，`application.json` 为 `application_complete`，`completed_calls=1`，
并核对 device_name、case_spec、安装对象哈希与本轮目标一致。查找输出文件：

```bash
find "$OUT" -type f \( \
  -name 'trace.json' -o \
  -name 'visualize_data.bin' -o \
  -name 'OpBasicInfo*.csv' \)
```

保留整个 `OUT` 目录。流水文件位于目标 kernel 的 `OPPROF_*` 子目录中，具体层级随工具版本变化。
本轮不执行 `--summarize`；它只统计 OpBasicInfo，不解析流水，也不需要为这次采集生成
`rgdr_task_summary.json`。当前工具若不支持 PipeTimeline，先核对工具版本与 A5 支持范围，
不要替换成 A2/A3 的 `TimelineDetail` 或 `MemoryDetail`。

### 如何查看与判读

1. 将生成的 `visualize_data.bin` 导入 MindStudio Insight，查看 Timeline 流水视图。
   `trace.json` 也应一并保留，便于后续分析。加载的是目标 RGDR 任务，而非输入构造任务。
2. 先看 Scalar 活动区与 Vector、MTE2、MTE3 的空隙和重叠：确认较长区间是持续控制活动，
   还是 Vector/搬运完成前的等待；汇总的 Scalar 时间不能直接当作某个函数或某次 Wait 的耗时。
3. 比较实际展示核的计算段长度、搬运先后和流水空隙，判断下一轮应针对元数据/地址控制，
   还是输入准备与搬运重叠。没有自定义标记时，流水不会自动标出 ProcessHead 等 C++ 函数名。

**该功能基于采样，不保证覆盖所有任务块。** 26.0 官方文档以仅展示 6 核描述覆盖限制；
本次 Ascend950DT_9582 的 AIV-only 结果实际导出 4 条 AIV 轨道，不能按固定 6 条验收。
Block Dim=32 而流水仅展示部分轨道，不表示其余核没有计算；
也不能保证采样包含上一轮 PMU 中时长最大的 block，或据此恢复所有 32 核的全局尾部。
若确需把区间对应到具体源码阶段，再单独设计可选 MarkStamp 标记；当前先保留原二进制进行采样。

PipeTimeline 属于诊断采集，其时延不替代第 2 节的 BasicInfo；原地更新 state 的重放输出不作精度验收。
参数、采样覆盖限制与可选 MarkStamp 的说明见官方
[msOpProf 使用说明：Pipeline Chart](https://github.com/Ascend/msopprof/blob/26.0.0/docs/en/user_guide/msopprof_user_guide.md#pipeline-chart)，
查看方式见 [MindStudio Insight 算子调优](https://gitcode.com/Ascend/msinsight/blob/26.0.0/docs/en/user_guide/operator_tuning.md)。
本机无 A5，以上命令尚待在目标环境执行；当前提交不声称已经生成真实流水。

## 7. 保留哪些结果

保留两组完整输出目录、各自的 `application.json`、BasicInfo 的 `rgdr_task_summary.json`，
以及 PMU 的全部目标 OpBasicInfo/PipeUtilization/MemoryUB CSV。另记 CANN/msprof 版本、设备型号、
构建安装命令和原始采集命令；不要只截取全核平均值。
PipeTimeline 另保留其完整输出目录，包含 application.json、trace.json、visualize_data.bin 和采集日志。
这份说明提供采集方法，本机没有 A5，尚未执行真实 PMU 采集。
