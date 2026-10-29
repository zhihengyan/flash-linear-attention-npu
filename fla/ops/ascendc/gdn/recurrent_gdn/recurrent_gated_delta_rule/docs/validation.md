# A5 VF 门控读取与同步实验开发记录

## 当前实验：第二步，删除专用路径无 Scalar 消费的显式 V_S

分支 `codex/gdn-a5-vf-sync-elision` 从 `ca44ab5b` 新建，保留第一步分支作为直接对照。
唯一主要变量是 ProcessSingleWork 中的显式 V_S Fetch/Set/Wait；不改 VF、Exp/Cast、队列、
两处 VF LocalMemBar、scratch 策略、GM 输出、host/key、公开接口、JSON、executor、标杆或容差。
case 188 按用户要求不处理。当前实现范围按 `03 → 04 → 05` 复核；沿用 CHUNK_DEPENDENT V1
参考（来源 commit a14961e9dfe8c8264768d67fb447101873da4ecf）的状态依赖/ready-free 闭环方法，
本算子仍为单 AIV Vector 路径，不移植参考的 Cube、跨核 flag 或 ping-pong 数量。

用户回传第一步与修改前性能几乎不变，未给出新的具体耗时、Task Duration 或本轮精度范围。
将第一步单独收益记录为未观测到，不能把未提供的精度报告补写为通过；fec1e18b 继续作为已保留基线。
本轮无设备，用户授权仅编码和静态检查，不请求 SSH；静态检查不能替代内网验收。

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A5 默认 perf 0，B=T=Nk=Nv=1，K=V=128，BF16 state，g-only | ca44ab5b 基线 / 第一步单独收益放弃 | VF 已读 UB 系数，Main Scalar 仍等待此前全部 Vector | 第一步读取迁移保留作第二步前提 | 本轮设备报告未提供 | 用户回传几乎持平；无具体新数值或 Task Duration | 未证明瓶颈变化 | 不替代 fec1e18b 的已保留结果 |
| 同上 | 第二步 / 待实测 | 无 UB Scalar 数值读取后，显式 V_S 仍可能阻塞 VF 发射 | 仅删除专用控制的显式 V_S Fetch/Set/Wait | 待静态检查和内网验证 | 待与 ca44ab5b 交错比较，再与 fec1e18b 比较累计结果 | 逻辑少一次 ID 取得和一对事件；不减少计算/访存字节数 | 正确性优先；精度失败恢复 ca44ab5b，无收益则结束该路线 |

### 依赖审计与风险

全部输入 MTE2 ready、输出 V→MTE3 ready/slot 释放继续由原 TQue 管理；本轮不删除这些事件。
gate Exp、beta/QKV/state Cast 和 q scale 的 UB 结果均由后续 Vector VF 消费；Main Scalar
只读取 GM metadata 和取得 UB 地址。special Compute 不再传入 gama_/beta_ 的标量数值。
第一步 VF 的两处 LocalMemBar 分别保护 gated scratch/dot 与第二遍读取、attention tmp 与输出转换，
必须保留。general 的标量路径原样保留。本轮无新增事件、buffer、workspace 或别名。

Stage 设计记录了每条依赖及官方同步说明：A5 PIPE_V 之间由硬件保序，VF 内 UB 读写流水仍需
LocalMemBar。官方 asc-devkit master 文档仅作硬件/API 语义依据；实际 CANN 9.1 的 Cast/Exp
lowering、VF 融合和自动同步须检查编译产物，不能把静态检查或 master 文档当作该版本的实测证明。
旧版 case 0 的非有限问题发生于仍有 UB Scalar GetValue 时；它说明同步敏感，不能推导所有
同步可删除，也不能推导本轮必然正确。若仍失败，先恢复 ca44ab5b 并保留非有限/MSS 报告。

### 验证计划与完成边界

本地检查：整个 arch35 header 与 ca44ab5b 归一化比较，只允许删除三条事件语句；
VF 和全部 Copy/Compute helper 保持一致，专用控制保留 metadata/空工作/地址/输入/输出顺序；
不能出现新的 Scalar UB GetValue、其它 barrier/event 或跨 pipe 替代。其余源码契约继续运行。
可选 C++ harness 只记录生产控制体的调用/地址，禁止将它描述为硬件完成模拟。

内网在同一 CANN/A5 环境，按下文目标算子构建/安装/产物一致性检查后，先执行：

```bash
ACCURACY_START=0 ACCURACY_END=1 ATK_GM_INIT_MODE=on \
  bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=accuracy --soc=ascend950
```

实际 case 0 必须命中 SINGLE_WORK=1，out/stateRef 有限且满足原精度标准；检查 ATK 结果而非仅退出码。
随后以原 200 条矩阵执行 accuracy，按原输入执行 MSS/确定性，覆盖 BF16/general、FP32/general。
需检查空工作、不合法 prefix/length 提前退出、非零 state index、padded block stride 和连续重复调用。
沿用既有输入资产；不重新生成 JSON，不为该实验改 seed、数据分布或容差。

精度与同步检查通过后，ca44ab5b/本轮交错运行相同 perf 0，沿用原预热、每轮 20 个 RGDR 样本、
统计与频率环境；只比较 RecurrentGatedDeltaRule 的 Task Duration，并单独记录端到端耗时。
最后与 fec1e18b 比较两步累计效果。不得将 18.335 μs 端到端值当作 RGDR kernel 基线。
目标是跨轮可重复且超过波动的下降；目前缺乏 V_S 独占成本，无法给出可信的数值收益下界。
反汇编核对专用实例的显式 V_S 被移除及 VF/输入 ready/输出 ready 的实现；其它 SDK/队列内部
仍可能发射 V_S，不能以整个 kernel 的 V_S 总数是否为零判定实验是否生效。

本轮本地源码回归 10 项：9 项通过，1 项因缺少 C++ 编译器跳过。整个 arch35 header 相对
ca44ab5b 的归一化差异仅为三条显式事件语句删除；VF/输入准备/队列/general/输出契约通过。
另外用内存中的六种错误变体验证该范围检查能拒绝：恢复 V_S、恢复 UB Scalar 读取、删除 VF
写读屏障、跳过 state 准备、删除输入 DeQue、丢失 state 输出。未将变体写入工作区。
Python AST、8 个本地 Markdown 链接及 `git diff --check` 通过；原精度/perf/MSS JSON、生成器、
executor 和 CPU 标杆与 ca44ab5b 一致。生产代码只修改 arch35 的 ProcessSingleWork。
这些结果只证明源码约束，本轮尚未执行 CANN 编译、A5 精度、MSS、确定性或性能。

## 第一步：保留 V_S 的 VF 门控读取实验

本轮直接基线 `fec1e18b`（`codex/gdn-a5-tail-scratch-elision`），
实验分支 `codex/gdn-a5-vf-gate-load`；历史 main 为 `c9e2f1db`。
用户授权仅编码和本地静态检查；A5/CANN 在内网，本地无 CANN、torch 或 C++ 编译器，不请求重新提供 SSH。
本文件保留到内网验收完成，不能把源码检查记为 NPU 精度或性能通过。

### 第一步实现与历史验证计划

用户授权从当前分支新建分支、实现方向 2 第一步并直接提交推送。当前工作区中上一轮结果记录
一并保留。本轮以 fec1e18b 为代码和性能对照，不把历史 4.48 μs 当作该父版本的 kernel 实测值。

唯一主要变量是门控系数的消费方式：原 Exp/Cast 继续生产 gamaInUb/betaInUb，专用路径不再
通过 Scalar GetValue 读取这两个 FP32 值，而将 UB 裸指针传给 VF。VF 复用一只 FP32 RegTensor，
在第一遍前广播 exp(g)、第二遍前广播 beta；对应三个系数乘法从 Muls 改为 Mul。
通用调用仍传两个 float、执行原 Muls；UB 指针模式仅在已有 SINGLE_WORK=1 实例可达。

V_S 的取得、Set/Wait 及位置保留；不实施第二步同步调整。两遍循环、两个 LocalMemBar、
尾部 scratch 免写回、队列、GM 输出、host/key/公开接口和全部测试 JSON/标杆/容差保留，case 188 不改。

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A5 默认 perf 0，BF16 state，g-only | fec1e18b 基线 | Scalar 从 UB 读两系数后再传入 VF | 已保留尾部 scratch 免写回 + V_S | 用户回传 200 条精度全过 | 端到端两次 18.34/18.33 μs，均值约 18.335 μs；kernel 未提供 | 无系数读取独占时间 | 本轮直接对照 |
| 同上 | 第一步 / 单独收益放弃 | 减少 Scalar 系数读取和交接，可能增加 VF 寄存器成本 | 仅迁移系数消费到 VF，保留 V_S | 本机静态检查；设备精度报告未提供 | 用户回传与修改前几乎持平，具体新数值未提供 | 两次 Scalar GetValue → 两次 VF 广播；逻辑读取字节数不减 | 未观测到单独收益；仅保留作第二步前提 |

验证计划：general 的 VF/Compute 展开后与父版本一致；专用路径仅替换两次读取和三个系数乘法，
广播位于各自 row 循环外，g/beta 地址及顺序正确；V_S、scratch 策略和其它生产/消费不变。
本机执行源码回归与可用的 C++ 检查，不能代替 CANN 编译或 A5 数值验证。内网先开启 GM 初始化
验证 case 0 和完整原精度矩阵，再验证 MSS/确定性及 scalar fallback，最后与 fec1e18b 比较
Task Duration 和端到端耗时。额外检查寄存器分配/spill；参数、预热、采样和统计沿用原 case。

本轮源码回归共 9 项：8 项通过，1 项因本机无 C++ 编译器跳过。主要检查包含：scalar VF 与
fec1e18b 一致；UB VF 仅增加两次广播、替换三个乘法；general Compute 保留；专用控制仅去掉
两次 Scalar GetValue；V_S、两遍计算、scratch 策略、host/key/公开接口和原测试输入保持。
本机没有 CANN 或 NPU，本轮未执行真实编译、精度、MSS 或性能，不把源码检查记为设备通过。
另验证源码检查能拒绝八种错误变体：衰减系数误读 beta、遗漏任一广播、beta 地址错误、
两系数指针交换、恢复额外 Scalar 读取、删除 V_S Wait、错误恢复尾部 scratch 写回。
Python AST、文档链接、修改范围和 `git diff --check` 通过；生产代码仅修改 arch35 私有实现。

## 已保留方向 1 fec1e18b 的实现与验证记录

用户明确授权以当前正确版本新建分支进行方向 1 编码。本轮只通过 VF 内部编译期参数
`writeUpdatedScratch` 控制第二遍 updated state 的 UB 写回；Compute 传 `!singleWork`。
原第一遍 gated state 写回、两遍循环、全部数值运算和 GM 输出、UB/queue/workspace、V_S 同步、
host/key/入口/接口保持。JSON、生成器、CPU 标杆、executor 和 case 188 均不修改。

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A5，B=T=Nk=Nv=1，K=V=128，BF16 state，g-only；perf 0 | c36e18ce 基线 | 每个 work 在 VF 第二遍仍写 updated scratch，后续无读取 | 已保留单 work + V_S | 用户报告精度全过 | Task Duration 两轮 4.51/4.45 μs，合并约 4.48 μs | 无同轮 Store/PMU | 本轮直接对照，不能用旧问题版本替代 |
| 同上 | 本轮 / 主 kernel 口径待实测 | 去掉无消费者的 UB Store，实际关键路径变化待 PMU | SINGLE_WORK=1 的 writeUpdatedScratch=false，其余 true | 用户回传 200 条精度全过 | 用户回传端到端两轮均值：父版本 18.500 → 本轮 18.335 μs，约改善 0.89%；主 kernel 待补 | 理论少写 UB 64 KiB/work；实际发射/重叠待核对 | 端到端结果支持保留候选；正式验收待完整证据 |

静态验证计划：解析 guarded store 的范围；true 分支应与原 VF 逐 token 一致，false 分支应仅少
第二遍 Store；Compute 的四种 gate dispatch 均由 `!singleWork` 控制；ProcessSingleWork 与 c36e18ce
一致；公开接口、host/key、测试资产、缓冲和同步保持。设备验证沿用下文单算子构建/安装/产物校验，
先 case 0 精度，再回归通用多 token case 2、FP32、gk、accepted、layout 及原有矩阵和 MSS/确定性。
原精度标准和值域不变；用户此前要求不处理的 case 188 保持原样。
本轮每次比较均重建、安装并校验对应分支产物，先精度后性能，每轮保持相同 20 个样本、预热与环境。
本机没有 CANN/NPU，不将父分支的设备验证结论转记为本实验通过。

本轮本地检查：`python -B tests/atk/recurrent_gated_delta_rule/scripts/test_single_work_fastpath.py -v`
共 8 项，7 项通过，1 项因无 C++ 编译器跳过。源码展开确认 true 与原 VF 相同、false 仅删除末尾
scratch Store；Compute/host/入口/控制路径和原输入资产的范围检查通过。此结果不能代替 CANN 编译、
硬件访存检查、精度或性能；本机未执行 NPU 验证，设备结果以下述用户回传为准。
另外验证了源码防回归检查能拒绝五种错误变体：实验未生效、general 错误省略写回、
错误删除第一遍 scratch、错误删除 GM state 输出、丢失 V_S Wait。Python AST、文档链接、
生产代码/测试输入修改范围和 `git diff --check` 均通过；这些均为静态证据。

### 用户回传：200 条精度与三分支性能对照

用户明确报告 200 个精度测试全过，并给出各分支两次性能执行时间。沿前文按端到端 μs
口径记录；未提供新分支 op_summary，因此不将这组 18 μs 级数据记为 RGDR Task Duration。

| 分支 | 第一次（μs） | 第二次（μs） | 两轮算术均值（μs） | 相对 main 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| main | 18.77 | 18.89 | 18.830 | — |
| gdn-a5-single-work-fastpath | 18.44 | 18.56 | 18.500 | 1.75% |
| gdn-a5-tail-scratch-elision | 18.34 | 18.33 | 18.335 | 2.63% |

本轮相对直接父版本减少约 0.165 μs（0.89%），相对 main 累计减少约 0.495 μs（2.63%）。
尾部 scratch 分支两次都快于父版本；这支持保留该候选，但只有两轮汇总，不能据 0.01 μs 的轮间差
宣称稳定性显著改善。没有原始日志、产物身份、轮次顺序和新分支主 kernel/PMU，不能把端到端
0.165 μs 全部归因为 Store 的独占时间，也不将旧实验 A 的退化归因于该子项。
建议后续以 fec1e18b 为正确候选基线；完整报告、MSS/确定性和统一主 kernel 口径仍待归档。

## 已保留基线 c36e18ce 的实现与验证记录

本节保留 c36e18ce 时的实现和判断；它仍通过 Scalar GetValue 消费系数。当前第二步的消费方和
显式同步已变化，以本文顶部实验记录及当前 design.md 为准，不能将本节通过结果转记到新分支。

设计见 [design.md](design.md)，接口边界见 [api.md](api.md)。改变单 work 标量控制、元数据复用，
并在 UB 标量读取前显式完成前置 Vector 操作。
不引入 A/B 或 profiling 分支代码，不更改 public ABI、CPU 标杆、200 条 ATK 精度用例或精度标准。
MSS / determinism 保留原 case 0，选取已有精度 case 1/2 补齐 FP32/general 与 BF16/general。

| 项目 | 证据 / 状态 |
| --- | --- |
| main 参考 | 用户 main/B 交错实验 main 均值 4.89、4.85、4.87 μs，合并 4.87 μs |
| 目标 | 相同原 case、环境、预热/采样方式下主 kernel 均值/中位数稳定下降；修复版两轮均值合并约 4.48 μs，相对历史 main 约改善 8.0%，同轮 main 待补 |
| 模板选择 | 声明 BF16/general、FP32/general、BF16/single 三个组合，入口传参与 host key 参数一致；源码检查通过，设备实际选择待验证 |
| VF/UB/同步 | 原 VF、InitLocalBuffers、Copy/Compute/ProcessHead/IsCurrentBlock 函数体及 general 调度与 main 相同；专用控制新增一对 V_S Set/Wait |
| 公共接口与原输入 | 原 ACLNN 声明/实现、OpDef、tiling struct 字段、CPU 标杆、200 条精度 JSON、perf JSON 和生成器对照 main 通过 |
| 本地脚本 | `test_single_work_fastpath.py -v`：6 项，5 通过，1 因无 C++ 编译器跳过 |
| 同步源码防回归检查 | 能拒绝原无同步版本、遗漏 Wait、Wait 位于标量读取之后、模板尖括号缺失四种错误变体；仅为源码检查 |
| 测试资产与文档 | Python AST、精度 200 / 性能 1 / MSS 3 条 JSON 与 case_spec 一致性、新增 Markdown 链接检查通过 |
| C++ 选择/控制语义测试 | 脚本会编译生产选择头和提取的单 work 控制函数；本机未运行，不算已通过 |
| 格式检查 | `git diff --check` 通过 |
| 实际 CANN 编译 | 本机未执行；用户已编译加 V_S 的专用路径并回传 case 0 成功 |
| ATK 精度 | 用户随后回传修复版精度测试全过；具体执行范围、seed 和完整报告待归档 |
| 非连续/空工作/accepted/gk/FP32/A2A3 回归 | 未执行设备验证 |
| 默认 case 性能与 PMU | 修复版用户回传 RGDR 均值 4.51/4.45 μs，每轮 20 样本，合并约 4.48 μs；端到端 18.4–18.5 μs，对照 main 18.7–18.8 μs；同轮 main kernel/PMU 待补 |

采用仓库 CHUNK_DEPENDENT/V1 参考（来源 a14961e9dfe8c8264768d67fb447101873da4ecf），
仅参考递推状态生命周期、tiling 模板与同步核对方法；本例没有 Cube 或跨核 workspace。

## 2026-10-29：case 0 非有限输出与 V_S 修复

固定原 case 0 和 seed=20260826，精度运行开启 `ATK_GM_INIT_MODE=on`，范围为 `[0,1)`。
以下设备结果均为用户回传，本机没有运行设备测试：

| 变体 | 结果 |
| --- | --- |
| 原专用路径 SINGLE_WORK=1 | out 的 128 个元素、stateRef 的 16384 个元素全部非有限 |
| 当前分支临时强制 singleWork=false | case 0 执行成功 |
| 恢复 SINGLE_WORK=1，在 LoadPrefetchedState 后、UB 标量读取前增加一对 V_S | case 0 执行成功 |

本轮将用户验证成功的 event 取得与 Set/Wait 原样加入专用路径，不保留强制 general 的诊断修改。
不改 JSON、生成器、executor、CPU 标杆或容差；case 188 按用户要求不处理。
结果支持前置 Vector 完成时序相关问题，尚不能单独判定是 gama、beta 或 state 输入准备的哪项依赖，
也不能据此宣称已经确认编译器缺陷。源码回归新增 Set/Wait 配对、发射位置及空工作提前退出检查；
可选 C++ harness 仅记录 API 发射顺序，不模拟设备完成或数值正确性，本机因无编译器而跳过。

用户后续回传：加 V_S 后精度测试全过，端到端耗时 18.4–18.5 μs，对照 main 的
18.7–18.8 μs（按用户所述区间理解）。区间差约 0.2–0.4 μs，中点 18.75 → 18.45 μs，
约改善 1.6%。这属于用户报告的端到端结果；不能替代修复版 RGDR Task Duration 或据此计算 V_S 的独占耗时。

用户继续提供两轮 RGDR Task Duration，并明确两轮都来自加 V_S 的修复版：

| 轮次 | 样本数 | 均值 / 中位数（μs） | 整体标准差（μs） | P90 / P99（μs） |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20 | 4.51 / 4.49 | 0.14 | 4.64 / 4.85 |
| 2 | 20 | 4.45 / 4.39 | 0.17 | 4.63 / 4.87 |

由相同样本数的轮均值汇总，修复版均值约 4.48 μs；相对历史 main 的 4.87 μs，
减少约 0.39 μs（8.0%）。未提供同轮 main，所以该数值目前是历史基线比较。
修复前问题版本均值 4.46 μs 与修复版相差约 0.02 μs，两轮修复版均值本身相差 0.06 μs，
现有汇总不足以证明 V_S 导致明显 kernel 退化，更不能将 0.02 μs 认作 V_S 的独占成本。
kernel 节省的 0.39 μs 在约 18.75 μs 的端到端尺度上粗略对应 2.1%，与端到端约 1–2% 的
改善量级一致；端到端先前变化的原因仍需同轮完整 task/调用链数据分析。

余下验证：归档修复版精度执行范围/报告，完成 MSS/确定性及单任务边界，并补充同轮
main/修复版的 RGDR Task Duration 交错统计。修复前约 8.4% 的 kernel 改善不能作为修复后的收益。
当前 V_S 位于 state Cast 之后，会等待该位置之前的 Vector 完成；将其前移到 gate/beta 之后
会改变输入搬运与 Vector 的重叠，并可能失去对 state 准备的保护，仅是待验证候选，未改动实现。
没有设备证据时不删除已使精度恢复的同步，也不将前移视为必然加速。

## c36e18ce 基线实现追溯与静态检查边界

| 设计项 | 实现位置 | 本地证据 / 剩余验证 |
| --- | --- | --- |
| A5 与 shape/gate/stride 条件 | `op_host/recurrent_gated_delta_rule_single_work.h`、tiling `DoLibApiTiling` | 三组合、入口参数对照通过；选择谓词的 C++ 执行和真实 tiling 日志待测 |
| 单 work、元数据复用 | arch35 `ProcessSingleWork` | 不含任务循环；prefix/length 先检查，state index 只读一次；行为测试待编译运行 |
| 前置 Vector → Scalar 完成依赖 | arch35 `ProcessSingleWork` 的 V_S Fetch/Set/Wait | 静态配对和顺序检查通过；用户报告精度全过、修复版均值约 4.48 μs；完整报告及同轮 main 待归档 |
| 回退调度 | arch35 `ProcessGeneral` | 函数体与 main `Process` 一致 |
| Stage 数学、搬运、队列和释放 | 原 VF、InitLocalBuffers、Copy/Compute | 函数体一致；生成机器码、自动同步及 NPU 精度待测 |
| 公开 ABI / tiling 数据 | 原 OpDef、aclnn、packed struct | 源码与 main 一致；新私有模板参数需重建 host 与 kernel，不能只替换旧 `.o` |
| 三个模板的 ATK 代表 | 精度 0/1/2、MSS 0/1/2；性能 0 | JSON 映射通过；实际 key/blockDim/命中证据待采集 |

无设备时只能确认编写的控制路径和原 helper 的源码边界。编译器是否裁掉常量分支、
自动同步是否改变、标量指令是否减少，必须从 A5 编译产物和 PMU 证明。
新专用分支的空工作、非零 state index、padded block stride 也须在设备上回归；
现有精度 200 条只在 case 0 覆盖新分支，不能把其它 shape 的回退验证当作这些边界的设备证据。

## 内网最小验证命令

已有当前环境 wheel 时，只需目标算子 run 包；本轮没有新 stable adapter 或接口，不必重编全部 wheel：

```bash
python -B tests/atk/recurrent_gated_delta_rule/scripts/test_single_work_fastpath.py -v
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
./build_out/fla_npu_linux-x86_64.run --install
python scripts/verify_libcust_opapi_md5.py --run-package build_out/fla_npu_linux-x86_64.run
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=mssanitizer
PERFORMANCE_START=0 PERFORMANCE_END=1 \
  bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=performance
```

run 包路径按实际架构选择；安装后使用新 Python 进程。确认产物比对实际包含目标 `.o`，不接受跳过比对。
范围是 `[start,end)`，默认 case 使用 0/1。精度必须有实际用例和通过证据，ATK 退出 0 不代替结果检查。
性能只取 OP Type=RecurrentGatedDeltaRule 的 Task Duration，不混入 Neg/Cast/有限性检查。
本轮以 ca44ab5b/新分支交错复测，再与 fec1e18b 比较；需要模板命中证据和 Scalar/Vector/MTE 分项，
不将 timer 分支 5.9341 μs 当作 main 或本轮基线。
每轮沿用此前 20 个 RGDR 样本；实际预热次数、ATK/CANN 版本和频率状态从内网配置/日志记录，
本机没有这些配置，不为它们填入推测值。
