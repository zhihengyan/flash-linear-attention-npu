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

## 性能用例

`atk_recurrent_gated_delta_rule_perf.json` 保留最小用例，并新增一条 `Hk=16、Hv=32` 的单 token
decode 用例。两条均使用 `K=V=128`、BF16 state、g-only、连续布局，不提供 accepted-token 输入。

| 性能 Case ID | 详细设计 ID | B / T | Hk / Hv | 来源与用途 |
|---:|---|---|---|---|
| 0 | RGDR-P001 | 1 / 1 | 1 / 1 | 原精度 case 0，保留最小开销对照 |
| 1 | RGDR-PERF001 | 1 / 1 | 16 / 32 | 从 case 0 派生，仅改变输入头数，用于观察多头 decode 性能 |

case 1 保留原用例的随机种子、输入生成方式、值域、状态索引和 gate 配置；同时更新用例编号、
名称和性能标签。它是用户指定头数的合成性能用例，不是现有精度用例的直接平移，也不是现网
抓取记录。原 200 条精度用例保持不变；case 1 的设备精度尚待验证，已回传的性能均值见开发记录。
原 single-work 模板仍要求 `Hk=Hv=1`。本分支把末 token scratch 省写推广到 A5 通用路径，
因此 case 1 的所有 head 也可以省去最后一次 updated FP32 scratch 写入。

只运行新增用例（范围为 `[1,2)`）：

```bash
PERFORMANCE_START=1 PERFORMANCE_END=2 \
  bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=performance
```

## 执行方式

### msprof op：直接测量性能 case 1

[msprof_op_rgdr.py](scripts/msprof_op_rgdr.py) 基于
`codex/gdn-a5-public-latency-diagnostic` 的同名脚本适配，默认 `--case-id=1`，
直接读取本目录性能 JSON 并复用 executor 的 `build_inputs`。可用 `--case-id=0` 测最小用例。
离线运行 `python scripts/msprof_op_rgdr.py --dry-run` 可核对输入配置，不加载 torch/NPU。

先在 Linux A5 上构建、安装并核对目标 run 包。激活 ATK/PyTorch 环境并加载 CANN 后，在仓根执行：

```bash
SCRIPT="$PWD/tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py"
OUT="$PWD/tests/atk/recurrent_gated_delta_rule/runs/msprof_op/multihead_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
msprof op \
  --application="python $SCRIPT --repo-root $PWD --device 0 --case-id 1 --calls 20 --metadata $OUT/application.json" \
  --kernel-name=RecurrentGatedDeltaRule \
  --launch-count=20 --warm-up=20 --aic-metrics=BasicInfo --output="$OUT"
python "$SCRIPT" --summarize "$OUT" --expected-count 20
```

`--calls`、`--launch-count` 和汇总期望样本数保持一致。脚本在首条目标调用前构造全部独立 state，
调用间只同步，不生成输入或重置 state。仅汇总 RGDR 的 `Task Duration(us)`；输入准备产生的
Cast/Neg 等任务不计入。汇总保留原始行，样本数不符或混入多个 run/device 时拒绝通过。
`application.json` 记录 case、源码提交、实际安装包、全部 OPP opapi 候选和 RGDR `.o` 哈希；
这些信息不能替代 `verify_libcust_opapi_md5.py` 对待测构建产物的核对。

先验证未被 profiler 重放的 case 1 精度，再进行计时。可在本算子目录运行原生 ATK：

```bash
cd tests/atk/recurrent_gated_delta_rule
atk node --name npu_dut --backend npu --devices 0 \
  node --name cpu_golden --backend cpu \
  task -c atk_recurrent_gated_delta_rule_perf.json --task accuracy --bm_device cpu \
  -p executor_recurrent_gated_delta_rule.py -s 1 -e 2
```

msprof 可能预热/重放原地更新 state 的 kernel，采集输出不用于精度验收。
当前采集基线为 `93f5c70a`；需要隔离末 token 省写时，可与 `7acfe2e1` 交错比较。
保留对应 run 包，使用同一脚本、JSON、设备和采集参数。
msprof 参数依据：[msOpProf 26.0 参数说明](https://github.com/Ascend/msopprof/blob/26.0.0/docs/en/user_guide/msopprof_user_guide.md)。
基于 93f5c70a 的 PipeUtilization、MemoryUB 命令、字段和判读方法见
[A5 指标采集说明](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/msprof-metrics.md)。
PipeTimeline 可直接复用此 Python 脚本，无需改 kernel：`--calls 1` 配合 msprof 的
`--launch-count=1 --warm-up=20 --aic-metrics=PipeTimeline`。完整命令、产物检查和部分核采样限制见
[PipeTimeline 采集步骤](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/msprof-metrics.md#6-pipetimeline采集并查看流水)。
本地离线测试见 `scripts/test_msprof_op_rgdr.py`；设计和待验证记录见
[design.md](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/design.md)、
[validation.md](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/validation.md)。

### 当前实验：A5 元数据复用

对照基线为 `a6d33dbb`，当前候选仅复用单序列长度和单 token 输出 state index。
性能 JSON、CPU 标杆、测评脚本与原 head 调度保持。候选尚待设备精度和同环境多轮性能验证，
不能以源码读取次数替代性能结论。构建安装、case 1 精度、原 200 条回归和交错测量步骤见
[元数据复用实验说明](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/metadata-reuse-experiment.md)。

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
