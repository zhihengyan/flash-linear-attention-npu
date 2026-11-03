# RecurrentGatedDeltaRule ATK 工程

本目录提供 `recurrent_gated_delta_rule` 的 ATK 单算子工程，包含 `executor_recurrent_gated_delta_rule.py`、`gen_recurrent_gated_delta_rule.py`、`recurrent_gated_delta_rule.yaml`、`atk_recurrent_gated_delta_rule.json`。测试通过稳定入口 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 调用算子。

## 输入约束

- `query/key` 的 shape 为 `(T, Nk, Dk)`，`value/out` 的 shape 为 `(T, Nv, Dv)`。
- `beta` 的 shape 为 `(T, Nv)`；`stateRef` 为原地输入输出，shape 为 `(BlockNum, Nv, Dv, Dk)`。
- `query/key/value/beta/out` 为 BF16；`stateRef` 支持 BF16 和 FP32。
- `actualSeqLengths` 为 `(B+1,)` 的 INT32 张量，首元素是不参与计算的无效前缀长度，所有元素之和等于 `T`。
- `ssmStateIndices` 为 `(T,)` 的 INT32 张量，取值范围为 `[0, BlockNum)`，且所有元素互不重复。
- `g` 如提供，shape 为 `(T, Nv)`；`gk` 如提供，shape 为 `(T, Nv, Dk)`。两者均为 FP32，且至少提供一个。
- `numAcceptedTokens` 如提供，shape 为 `(B,)`，每项取值范围为 `[1, actualSeqLengths[i+1]]`。
- `Nv` 必须是 `Nk` 的整数倍；`Nk <= 256`、`Nv <= 256`、`Dk=128`、`Dv=128`；每条有效序列长度不超过 8。
- 泛化矩阵覆盖连续 state、transpose 非连续 state、head padding、block padding、复合 padding 和 storage offset；同时覆盖 Q/K/V/gate/metadata 的合法非连续视图。

## 标杆来源

`tests/atk/recurrent_gated_delta_rule/reference_recurrent_gated_delta_rule.py`；`fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/README.md`

CPU 标杆在本目录的 `reference_recurrent_gated_delta_rule.py` 中实现，输入构造、`run_cpu`、`run_npu` 和 `FunctionApi` 在 `executor_recurrent_gated_delta_rule.py` 中实现；公共文件只提供基础工具函数。CPU golden 与算子语义一致，按 batch 分段使用 FP32 执行递推，覆盖 GQA 头映射、`g/gk` 衰减、接受 token 对初始状态索引的选择，以及逐 token 状态写回。精度检查统一使用 ATK 原生 `mixed_tolerance_bm`。

ATK 比较以下两个结果：

- `out`：`(T, Nv, Dv)` 的注意力输出；无效前缀没有公开数值语义，比较前统一置零。
- `stateRef`：算子执行后原地更新的完整状态张量。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。提交的 JSON 用例使用 `soc=all`，同一矩阵用于 A2、A3、A5。

## 默认用例

`atk_recurrent_gated_delta_rule.json` 由 `gen_recurrent_gated_delta_rule.py` 确定性生成，固定包含 200 条正向用例。每条 `case_spec` 都携带详细设计 `design_id`，映射关系如下：

| ATK Case ID | 详细设计 ID | 数量 | 覆盖范围 |
|---:|---|---:|---|
| 0-5 | RGDR-P001-P006 | 6 | gate 组合、BF16/FP32 state、A2/A3 归约路径 |
| 6-21 | RGDR-P007-P022 | 16 | 变长、prefix、零长度逻辑序列、accepted token |
| 22 | RGDR-P023 | 1 | `Dk=Dv=128` 的对齐维度、UB profile |
| 23-38 | RGDR-P024-P039 | 16 | `Nk/Nv=1..256`、GVA group size |
| 39-54 | RGDR-P040-P055 | 16 | state stride/layout、稀疏/逆序/置换 index |
| 55-69 | RGDR-P056-P070 | 15 | 平台、调用通路代表 shape、非连续输入、连续调用与回归 |
| 70-81 | RGDR-P071-P082 | 12 | gate、state dtype/layout 与非连续输入交叉组合 |
| 82-90 | RGDR-P083-P091 | 9 | 长变长链、accepted token、state index 置换与 scale |
| 91-95 | RGDR-P092-P096 | 5 | 补充可整除的 `Nk/Nv` 头映射 |
| 96-99 | RGDR-P097-P100 | 4 | 连续调用与 gate/beta 数值分布 |
| 100-115 | RGDR-G001-G016 | 16 | GVA head 映射与 group size |
| 116-131 | RGDR-G017-G032 | 16 | GVA gate、state dtype、可追踪 head 数据 |
| 132-147 | RGDR-G033-G048 | 16 | GVA varlen、prefix、accepted 与稀疏 block |
| 148 | RGDR-G049 | 1 | GVA 对齐维度、UB 分片（`Dk=Dv=128`） |
| 149-164 | RGDR-G050-G065 | 16 | GVA state layout、index、非连续输入与复杂视图 |
| 165-174 | RGDR-G066-G075 | 10 | 补充 GVA ratio 与大 head 边界 |
| 175-182 | RGDR-G076-G083 | 8 | GVA gate、state layout 与数值 profile 交叉 |
| 183-190 | RGDR-G084-G091 | 8 | GVA 长变长、accepted token 与 state index 链 |
| 191-198 | RGDR-G092-G099 | 8 | GVA 复合非连续视图与连续调用 |
| 199 | RGDR-G100 | 1 | prefix、varlen、accepted、GVA 与复合 layout 回归 |

ATK DUT 统一通过 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 执行。`design_routes` 只保留详细设计中 P/A/D/F 的来源追溯，不表示本 JSON 已执行 aclnn C++ example 或快速拉起通路；这些通路仍由各自专用测试验证。

正向 JSON 只包含 `Dk=Dv=128` 的合法 shape。任何不等于 128 的维度属于非法输入，由 `fla_npu` wrapper 和 op_host tiling 共同拦截；对应拦截由 op_host UT 覆盖。

## A5 单 work 实验的模板覆盖

本分支仅对 A5、BF16 state、B=T=Nk=Nv=1、K=V=vStep=128、blockDim=1、
stateStride2=128、g-only、无 accepted-token 输入选择 `SINGLE_WORK=1`。
其它合法输入选择 general；A2/A3 始终选择 general。模板详设见
[design.md](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/design.md)。

| 模板参数 `(D_T_STATE,SINGLE_WORK)` | 精度 case | perf case | MSS / determinism case | 设备选择证据 |
| --- | ---: | ---: | ---: | --- |
| `(BF16,1)`，A5 single | 0 | 0 | 0 | 用户历史报告 scratch 200 条精度通过；新增诊断脚本未上机 |
| `(FP32,0)`，general | 1 | — | 1 | 内网待采集 |
| `(BF16,0)`，general | 2 | — | 2 | 内网待采集 |

`_mss.json` 保留原 case 0，人工选取原精度 case 1/2 补齐 general 的两个模板，
不改变原 200 条精度用例、perf case 0、CPU 标杆或容差。
数值 TilingKey 应以当前 CANN 生成结果为准，不能直接沿用 main 的历史数值。
host DEBUG 日志新增 `stateDtypeKey / singleWork / tilingKey`，与实际 kernel 选择及
blockDim 对照；不能只根据 case 的 shape 认定命中。

已保留基线 `c36e18ce`：2026-10-29，专用路径在 `LoadPrefetchedState` 后、读取 UB 中的 g/beta 前增加一对
`V_S` Set/Wait。用户回传 GM 初始化开启时 case 0 成功，随后报告精度测试全过；
具体执行范围和报告待归档，MSS/确定性待验证。修复后端到端为 18.4–18.5 μs，
对照 main 18.7–18.8 μs。修复版 RGDR 两轮各 20 个样本，均值分别为 4.51/4.45 μs，
合并约 4.48 μs，相对历史 main 4.87 μs 约改善 8.0%；同轮 main 的 kernel 统计待补。
修复前的性能结果不能作为本版本的收益结论。

### 已保留基线：尾部 FP32 scratch 免写回

分支 `codex/gdn-a5-tail-scratch-elision` 从 c36e18ce 创建，只让 SINGLE_WORK=1 的 VF 使用
`writeUpdatedScratch=false`，省去第二遍更新后的 64 KiB UB 写回。第一遍 gated state、
两遍循环、V_S、GM state/out 和缓冲配置保留；general 的多 token 递推仍写 updated scratch。
该开关不改变 host/key、公开接口或测试输入，精度/性能/MSS JSON 均与父版本一致，case 188 不改动。
本轮性能直接与 c36e18ce 的相同 perf case 0 交错比较，不将父版本的通过记录当作本实验结果。

无需 NPU 的本地回归入口如下；没有 C++ 编译器时会明确跳过 C++ 控制语义测试，
其余源码对照不能代替 CANN 编译、模板选择、NPU 精度或同步检查：

```bash
python -B tests/atk/recurrent_gated_delta_rule/scripts/test_single_work_fastpath.py -v
```

该脚本核对原缓冲/搬运/同步函数和公开接口与 main 基线一致，并将 VF 的 scratch guard 作源码展开：
true 应与 c36e18ce 原 VF 一致，false 应仅删去第二遍尾部 Store；四个 gate dispatch 均由
`!singleWork` 控制，ProcessSingleWork 与父版本一致。该检查不是 CANN 编译或设备数值验证。
有 C++17 编译器时，
还会执行生产代码中的选择谓词和单 work 控制函数，检查空工作、不合法 prefix/length、
非零 state index 和 padded block stride 的地址及调用顺序。专用路径的新增 V_S 检查
覆盖 Set/Wait 配对、标量读取前完成 Wait，以及空工作不发射事件；录制 API 顺序不等于验证硬件同步。
内网构建、安装、产物校验和 A5 验证步骤见
[开发记录](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/validation.md)。

## 固定输入的公开入口延迟诊断

`codex/gdn-a5-public-latency-diagnostic` 从 scratch `fec1e18b` 创建，仅增加诊断工具。
kernel/host/runtime 与 scratch 一致，不包含后续 state Cast 或输入启动融合实验。
原 ATK JSON/YAML、executor、生成器、标杆和容差保持。本节是独立诊断，正式验收仍走 ATK。

2026-11-02 用户回传的 ATK `device_perf=18.27275 us` 是 20 次调用中全部 task 的平均总和：
Neg 1.96665、RGDR 4.42955、out 有限性检查 5.50975、state 有限性检查 6.36680 us。
它包含测试输入生成和检查，不能当作公开入口墙钟延迟。原日志未提供代码 commit，不推定其源码版本。

### 第一步：在 scratch 诊断分支构建并运行

在已激活的原 CANN/conda 环境、仓库根目录执行。下例设备 1 与上述日志一致；两版本使用同一设备。
只需重建单算子 run 包，无需全量 wheel。本脚本会检查 `.run` 的 op_api 与实际包路径，以及
当前 build 和已安装 OPP 的 RGDR `.o` 相对路径/哈希集合；缺失、跳过或不匹配均不会当作通过。

```bash
git fetch origin
git switch codex/gdn-a5-public-latency-diagnostic
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
./build_out/fla_npu_linux-x86_64.run --install

python tests/atk/recurrent_gated_delta_rule/scripts/measure_public_latency.py \
  --device 1 --case-id 0 --label scratch \
  --run-package build_out/fla_npu_linux-x86_64.run \
  --warmup 20 --samples 100 --profile --profile-samples 20
```

输入来自原 perf case 0 的 `build_inputs`，只生成一次；每次调用前恢复同一个 state 张量并同步，
恢复和预同步在计时区间外。该协议会影响缓存状态，绝对时间不与旧 ATK 结果混算。
两版均在 `torch.inference_mode()` 下运行。host 和 Event 独立循环；输出在完成同步前保持引用，
不在窗口内 D2H、检查有限性或重新构造输入。
输出/最终 state 的有限性检查放在整个循环之后，仅为 sanity check，不宣称设备精度通过。

| 输出字段 | 范围与解释 |
| --- | --- |
| `host_submit_us` | Python 公开调用（含薄调用闭包）开始到返回；不等待设备完成，不等同 kernel launch 时间 |
| `call_and_sync_us` | 同一次公开调用开始到当前 stream 同步返回；含 host、等待和同步 API 开销 |
| `post_return_wait_us` | 调用返回到同步返回；含队列/设备等待和同步开销，不是纯设备执行时间 |
| `stream_event_us` | 独立循环中调用前后两个 NPU Event 的区间；可能包含 host 下发导致的 stream 空闲 |
| `empty_*` | 同协议的空调用/空同步/Event 控制，单独报告，不自动相减 |
| `system_profile` | 计时结束后单独采集的原始系统 profile；RGDR `Task Duration(us)` 才是主 kernel 指标 |

Event 的 `elapsed_time` 单位为 ms，脚本转为 us。接口依据见
[torch_npu Event/Stream](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/streams.py)。
系统 profile 按官方 [profiler 实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/profiler.py)
和 [使用示例](https://gitee.com/ascend/mstt/tree/master/profiler) 接入；本机尚无 NPU 运行验证。
Event、host 墙钟和 Task Duration 不互相替代；同步后逐次调用是串行诊断，不是模型吞吐测试。

`--profile` 额外执行 1 次 profiler warmup 和默认 20 次 active 调用，
使用有限的 schedule（wait=0、warmup=1、active=profile-samples、repeat=1），
最后一次 step 完成 RECORD_AND_SAVE 后退出，避免在默认 RECORD 状态停止的警告。
Level0 显式设置 `AiCMetrics.AiCoreNone`，不请求硬件流水指标。先在 profiler 外预分配相同初值的独立 state，
采集窗口内不做 state reset、Neg 或有限性检查，保留其它真实调用任务。设置 `data_simplification=False`
保留 CANN 导出的 op_summary，避免精简阶段删除原始任务表。同步/CPU marker 会扰动间隙，
profile 墙钟与无 profiler 的统计分开。兼容原 `op_summary*.csv`（OP Type / Task Duration(us)）
和 torch_npu 2.10 的 `kernel_details.csv`（Type / Duration(us)）。后者由官方逐列映射自前者，
不是用 Event 或 wait 时间替代 Task Duration。参考
[官方列名映射](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/profiler/analysis/prof_common_func/_csv_headers.py)。
用户 A5 实际导出中 Type/OP Type 均为 N/A，但 Name/Op Name 是完整的
`aclnnRecurrentGatedDeltaRule_RecurrentGatedDeltaRule_RecurrentGatedDeltaRule`。
类型缺失时，仅用该完整名称精确回退；不做子串匹配、不覆盖明确的其它类型，继续读取原 Duration。
inventory 保留原类型计数、总行数和 `rgdr_match_sources`。task_time 中的 PLACE_HOLDER_SQE 不计入 RGDR。
每份任务表独立校验目标 task 数；不拼接 partial 文件、不累加重复导出视图。至少有一份完整且计数
正确的任务表才通过；partial/unsupported 表的原因保留在 `profile_inventory.json`。
每次 NPU 测量有 60 秒 watchdog（含准备/下发/同步）；超时会终止进程，留下已有结果。

脚本打印 `output=...`。默认结果位于本算子 `runs/public_latency/<UTC时间戳>/`，包含 `result.json`、
`artifact_check.log` 和可选 `profile/`、`profile_inventory.json`。记录 commit/dirty、设备/stream、实际包/库/目标 `.o` 哈希、
case_spec 和原始计时样本。已有输出目录会拒绝覆盖；失败状态不会记为成功。

### 已完成计时、汇总 profile 失败时的恢复

若终端已打印全部 host/Event 统计，但最后报 `no RGDR op_summary ...`，先重解析现有数据，
无需重建/安装 `.run`、无需 NPU，也不重复 100 次计时。先更新本诊断分支中的脚本，再在仓根执行：

```bash
python tests/atk/recurrent_gated_delta_rule/scripts/measure_public_latency.py \
  --parse-profile tests/atk/recurrent_gated_delta_rule/runs/public_latency/20261102T122553_836006Z/profile \
  --profile-samples 20
```

把路径替换为对应运行目录即可。解析结果写入该路径下新建的 `reparse/<UTC时间戳>/`，
包括 `result.json` 和 `profile_inventory.json`，不会改写原运行的 result.json/统计。
成功时回传 `profile:` 行。若仍失败，回传新 `profile_inventory.json` 的文本，
其中包含实际 CSV 文件名、表头、目标任务数和拒绝原因；不能以缺失或部分数据宣称 profile 成功。

只有旧采集缺少完整任务表时，才重新采集 profile；不必重复 host/Event 计时：

```bash
python tests/atk/recurrent_gated_delta_rule/scripts/measure_public_latency.py \
  --device 1 --label scratch --run-package build_out/fla_npu_linux-x86_64.run \
  --profile-only --profile-samples 20
```

`--profile-only` 仍核对安装产物、生成原输入并预热，不产生新的 host/Event 样本。
在线模式在计时已完成而 profile 失败时标记 `timings_complete_profile_failed`，保留原样本和统计。
2026-11-02 用户回传 scratch 的 host_submit 中位数 11.738 us、call_and_sync 中位数 23.711 us，
均值分别 12.41849/26.14316 us，存在长尾；Event 中位数 16.881 us。它们是有效的未加 profiler
诊断样本，不能将 Event 时间称作 RGDR kernel 时间，也不能在缺少同协议 main 数据时宣称提速。

### 第二步：用同一脚本测 main

切换前把诊断脚本复制到 `/tmp`；main 本身不需要合入或修改诊断代码。切换后重建、安装 main 的
run 包，`--repo-root` 指向 main checkout，脚本会读取该 checkout 的原始输入生成与用例。

```bash
cp tests/atk/recurrent_gated_delta_rule/scripts/measure_public_latency.py /tmp/rgdr_measure_public_latency.py
git switch main
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
./build_out/fla_npu_linux-x86_64.run --install

python /tmp/rgdr_measure_public_latency.py --repo-root "$PWD" \
  --device 1 --case-id 0 --label main \
  --run-package build_out/fla_npu_linux-x86_64.run \
  --warmup 20 --samples 100 --profile --profile-samples 20
```

先回传两组 stdout 的统计行和 `profile:` 行即可，无需压缩包或全量 trace。
关注 `host_submit_us / call_and_sync_us / stream_event_us` 与 RGDR Task Duration，
确认瓶颈后再选 kernel、调用层或任务消除方向。原 ATK 指标继续保留，不通过移除检查宣称提速。

本地离线检查入口（不加载 torch/NPU）：

```bash
python -B tests/atk/recurrent_gated_delta_rule/scripts/test_measure_public_latency.py -v
```

2026-11-02 类型 N/A 修复后的本地控制/证据检查 14 项通过；原 scratch 检查记录为 7 项通过、
1 项因无 C++ 编译器跳过。
这些检查使用录制回调/文件 fixture，不执行 torch/NPU，也不模拟硬件计时或验收设备精度。

## msprof op 直调与设备任务耗时统计

独立脚本 [msprof_op_rgdr.py](scripts/msprof_op_rgdr.py) 复用原 `_perf.json` case 0 和
executor 的 `build_inputs`，通过公开 Python 入口直接调用 RGDR，不经过 ATK `FunctionApi`。
用例、随机种子、值域、CPU 标杆和容差不变。本入口是单算子性能诊断，不替代 ATK 正式验收。

在 Linux A5 上激活原 ATK/PyTorch 环境、加载 CANN 9.1 环境，先安装待测分支构建的产物。
脚本使用已安装的 `fla_npu`，不自动编译或安装；`application.json` 记录实际包、opapi、RGDR `.o`
哈希及源码版本，但不承担当前 build/run 包与安装产物的等价校验。两分支对比前分别重建安装。

在仓根目录执行（设备 1 可按需替换；不要和其它采集任务共用同一设备）：

```bash
SCRIPT="$PWD/tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py"
OUT="$PWD/tests/atk/recurrent_gated_delta_rule/runs/msprof_op/scratch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

msprof op \
  --application="python $SCRIPT --repo-root $PWD --device 1 --case-id 0 --calls 20 --label scratch --metadata $OUT/application.json" \
  --kernel-name=RecurrentGatedDeltaRule \
  --launch-count=20 --warm-up=20 --aic-metrics=BasicInfo \
  --output="$OUT"

python "$SCRIPT" --summarize "$OUT" --expected-count 20
```

`--kernel-name` 使用 RGDR kernel 名前缀过滤，避免选中输入生成产生的 Neg 等任务。
`BasicInfo` 只采集算子基本信息；预热由 msprof 的 `--warm-up` 控制，Python 不额外下发 RGDR
预热任务，避免将 Python 预热误计为正式样本。`--calls`、`--launch-count` 和汇总期望数量保持一致。
输入只构造一次，每次调用使用预先克隆好的独立 state；全部克隆在第一条 RGDR 之前完成。
调用间只同步，不生成输入、恢复 state、运行 CPU 标杆或有限性检查。保留输出至任务完成。
工具可能对原地更新 state 的 kernel 预热/重放，因此本脚本不对其输出进行精度验收。

汇总只读 `OpBasicInfo*.csv` 的 `Task Duration(us)`，输出均值、中位数、整体标准差、CV、
min/max、P50/P90/P99；完整样本、原始行和 CSV 来源保存在 `OUT/rgdr_task_summary.json`。
不将 Python 墙钟、Event、逐核时间或其它算子耗时混入该统计。Task Duration 包含设备任务调度、
执行与完成响应，不应称为纯算术时间。未知列名/单位、无目标记录、缺样本、重复导致超数、
多个 OPPROF 运行目录或多设备都会失败，并保留 inventory/error，不能把缺失数据当作成功。

官方参数和字段依据：
[msOpProf 26.0 参数说明](https://github.com/Ascend/msopprof/blob/26.0.0/docs/en/user_guide/msopprof_user_guide.md)、
[OpBasicInfo 字段](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md#opbasicinfo-basic-operator-information)。
msprof op 和此前 torch_npu profiler 的采集协议不同；只在相同工具、选项、设备和安装条件下做
main/scratch 对比，不把新结果与历史 ATK/torch_npu profile 直接拼接比较。

对比 main 时先复制这一份脚本，再切分支重建安装，在 main 仓根目录重复上述命令；将 `SCRIPT`
改为 `/tmp/msprof_op_rgdr.py`、输出目录和 `--label` 改为 main。脚本不依赖本分支的其它诊断脚本。

```bash
cp tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py /tmp/msprof_op_rgdr.py
```

离线检查（不加载 torch/NPU）：

```bash
python -B tests/atk/recurrent_gated_delta_rule/scripts/test_msprof_op_rgdr.py -v
```

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -scope=gen_cases
```

提交的 200 条 JSON 可通过以下命令重新生成并复核：

```bash
python3 tests/atk/recurrent_gated_delta_rule/gen_recurrent_gated_delta_rule.py --summary
```

统一脚本的 `gen_cases` 默认传入 `-dt 100 -en 0`。需要通过 ATK case generator 生成完整矩阵时，显式设置：

```bash
GEN_CASES_DTYPE_NUMBERS=200 \
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -scope=gen_cases
```
