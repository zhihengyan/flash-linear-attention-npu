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
| `(BF16,1)`，A5 single | 0 | 0 | 0 | fec1e18b 用户回传 200 条精度全过；本轮同步调整待验证 |
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

### 已保留方向 1：尾部 FP32 scratch 免写回

分支 `codex/gdn-a5-tail-scratch-elision` 从 c36e18ce 创建，只让 SINGLE_WORK=1 的 VF 使用
`writeUpdatedScratch=false`，省去第二遍更新后的 64 KiB UB 写回。第一遍 gated state、
两遍循环、V_S、GM state/out 和缓冲配置保留；general 的多 token 递推仍写 updated scratch。
该开关不改变 host/key、公开接口或测试输入，精度/性能/MSS JSON 均与父版本一致，case 188 不改动。
本轮性能直接与 c36e18ce 的相同 perf case 0 交错比较，不将父版本的通过记录当作本实验结果。

用户回传本轮 200 条精度全过。三分支各两次性能执行时间均值为 main 18.830 μs、
single-work 18.500 μs、tail-scratch 18.335 μs；尾部免写回相对父版本约改善 0.89%，
相对 main 累计约改善 2.63%。此处沿前文记录端到端耗时，新分支 RGDR Task Duration、
原始运行报告及 MSS/确定性仍待归档，详见开发记录。

### 第一步：VF 门控读取，保留 V_S

分支 `codex/gdn-a5-vf-gate-load` 从 fec1e18b 创建，仅对 SINGLE_WORK=1 迁移系数消费：
Exp/Cast 仍在原位置，原 V_S 的位置和 Set/Wait 保留；VF 从 gamaInUb/betaInUb 地址分别广播
加载 exp(g) 和已转换的 FP32 beta。复用一只寄存器，第一遍用衰减系数，第二遍用 beta，
以三个 FP32 Mul 替代原 Muls。general 继续传 float、执行原 Muls；两遍 VF、尾部免写回及 GM 输出保留。
host/key/接口、所有 JSON、CPU 标杆、executor、容差及 case 188 均不改变。

用户回传第一步 ca44ab5b 与修改前性能几乎不变，未提供具体新耗时、Task Duration 或精度范围。
记录为未观测到单独收益，仅保留作第二步前提；继续以 fec1e18b 为已保留性能基线。
fec1e18b 的 18.335 μs 是端到端参考，不能当作第一步或本轮的 kernel 基线。

### 当前实验：第二步，删除无 Scalar 消费的显式 V_S

分支 `codex/gdn-a5-vf-sync-elision` 从 ca44ab5b 新建，仅删除 ProcessSingleWork 的显式
V_S Fetch/Set/Wait。第一步 VF 读 UB 的实现、各 Exp/Cast、队列同步、两处 VF LocalMemBar、
两遍循环、尾部 scratch 免写回和 GM 输出保持不变；general/host/key/接口及所有测试输入均不改。
原先 V_S 保护的 UB Scalar GetValue 已在第一步删除；输入准备结果现在均由 Vector VF 消费。
依赖审计见设计与开发记录，不能仅以源码顺序当作 CANN 9.1 设备验证。

内网先开启 GM 初始化跑 accuracy case 0，确认 out/stateRef 有限且满足原精度规则，再运行原完整
accuracy 矩阵、MSS 和确定性。若重现 NaN/Inf 或同步错误，恢复 ca44ab5b 并保留日志。
精度通过后与 ca44ab5b 交错比较 RGDR Task Duration/端到端耗时，隔离同步效果；
再与 fec1e18b 比较两步累计结果。沿用相同输入、预热、每轮 20 个 RGDR 样本和统计方法。
本轮无设备验证或性能结果；case 188 不作修复，JSON、标杆、executor、值域、seed 和容差均不改变。

无需 NPU 的本地回归入口如下；没有 C++ 编译器时会明确跳过 C++ 控制语义测试，
其余源码对照不能代替 CANN 编译、模板选择、NPU 精度或同步检查：

```bash
python -B tests/atk/recurrent_gated_delta_rule/scripts/test_single_work_fastpath.py -v
```

本轮本地共 10 项检查：9 项通过，1 项因无 C++ 编译器跳过；本机无 CANN/NPU，未执行设备验证。

该脚本核对原缓冲/搬运函数和公开接口与 main 基线一致。将门控读取方式作源码展开后，
scalar 模式应与 fec1e18b VF 一致，UB 模式应仅增加两次循环外广播并替换三个系数乘法；
general Compute 与父版本一致。整个 arch35 header 相对 ca44ab5b 只允许删除专用控制的三条
V_S Fetch/Set/Wait 语句（注释忽略）；全部输入 helper、VF 数学、两处 LocalMemBar 和输出队列保持。
scratch guard 的 true/false 展开继续检查通用递推与尾部免写回范围。该检查不是 CANN 编译或设备数值验证。
有 C++17 编译器时，
还会执行生产代码中的选择谓词和单 work 控制函数，检查空工作、不合法 prefix/length、
非零 state index 和 padded block stride 的地址及调用顺序。专用路径检查全部准备调用在 Compute
之前，空工作不搬入/发射 VF，且没有 Scalar UB 读取或额外事件；录制 API 顺序不等于验证硬件同步。
内网构建、安装、产物校验和 A5 验证步骤见
[开发记录](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/validation.md)。

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
