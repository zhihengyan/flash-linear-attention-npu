# A5 单 work 与尾部 scratch 实验开发记录

本轮直接基线 `c36e18ce`（`codex/gdn-a5-single-work-fastpath`），
实验分支 `codex/gdn-a5-tail-scratch-elision`；历史 main 为 `c9e2f1db`。
用户授权仅编码和本地静态检查；A5/CANN 在内网，本地无 CANN、torch 或 C++ 编译器，不请求重新提供 SSH。
本文件保留到内网验收完成，不能把源码检查记为 NPU 精度或性能通过。

## 当前实验：仅省去单任务尾部 FP32 scratch 写回

用户明确授权以当前正确版本新建分支进行方向 1 编码。本轮只通过 VF 内部编译期参数
`writeUpdatedScratch` 控制第二遍 updated state 的 UB 写回；Compute 传 `!singleWork`。
原第一遍 gated state 写回、两遍循环、全部数值运算和 GM 输出、UB/queue/workspace、V_S 同步、
host/key/入口/接口保持。JSON、生成器、CPU 标杆、executor 和 case 188 均不修改。

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A5，B=T=Nk=Nv=1，K=V=128，BF16 state，g-only；perf 0 | c36e18ce 基线 | 每个 work 在 VF 第二遍仍写 updated scratch，后续无读取 | 已保留单 work + V_S | 用户报告精度全过 | Task Duration 两轮 4.51/4.45 μs，合并约 4.48 μs | 无同轮 Store/PMU | 本轮直接对照，不能用旧问题版本替代 |
| 同上 | 本轮 / 待实测 | 去掉无消费者的 UB Store，是否在关键路径尚待实测 | SINGLE_WORK=1 的 writeUpdatedScratch=false，其余 true | 本机仅静态检查，设备待测 | 待与 c36e18ce 交错比较 | 理论少写 UB 64 KiB/work；实际发射/重叠待核对 | 未宣称性能收益 |

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
硬件访存检查、精度或性能；本轮 NPU 验证仍未执行。
另外验证了源码防回归检查能拒绝五种错误变体：实验未生效、general 错误省略写回、
错误删除第一遍 scratch、错误删除 GM state 输出、丢失 V_S Wait。Python AST、文档链接、
生产代码/测试输入修改范围和 `git diff --check` 均通过；这些均为静态证据。

## 已保留基线 c36e18ce 的实现与验证记录

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

## 实现追溯与静态检查边界

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
main/C 交错复测；需要模板命中证据和 Scalar/Vector/MTE 分项，不将 timer 分支 5.9341 μs 当作 main 基线。
每轮沿用此前 20 个 RGDR 样本；实际预热次数、ATK/CANN 版本和频率状态从内网配置/日志记录，
本机没有这些配置，不为它们填入推测值。
