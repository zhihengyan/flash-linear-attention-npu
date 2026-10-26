# A5 实验 A 复盘与实验 B 开发记录

## 基线和证据边界

- main：`571d7acf`；A：`5c5947d7`；B 分支：`codex/gdn-a5-ub-buffer-profile`，直接从 main 建立。
- 用户提供 A5 结果：main 18.7685 μs，A 19.3954 μs，差值 +0.6269 μs（+3.34017%）。
  本地未找到所指的 `atk_output/perf/**/atk.log` 或 op_summary/指令 profiling，故保留为用户报告值。
- 当前 perf JSON case 0 是 B=T=Nk=Nv=1、K=V=128、g-only/BF16 state。未获得原日志前，
  不能确认所报时间是否正是此 case、是否为单 kernel Task Duration，或是否已排除测量波动。
- A 减少 VF 内 UB 流量，GM 流量、Cast 和调用链不变；行内融合引入连续的
  ReduceSum→Duplicate→delta→update→第二次 ReduceSum 依赖，并延长 state 活跃期。
  调度/双发射变差或寄存器压力是候选原因，尚无 spill 或周期占比证据。
  需同环境交替运行 main/A、多轮统计，并比较指令、寄存器/spill、VF 与 MTE 等待才能归因。
- A5 的 VF 调度依据见官方
  [双指令发射优化](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/operator_practice/simd_operator_optimization/vector_compute/vf_optimization/dual_instruction_issue_optimization.md)。
  此文支持依赖链会影响调度的机制解释，不是本次回退根因的证据。
- 用户此前已指定内网执行设备验证，本地仅编码和静态检查；本轮沿用该范围，不声明设备验收通过。

## 实验追踪

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 用户报告，原始日志未在本地 | 基线 | main 两遍 VF | 原共享 UB 策略 | 未提供本轮精度明细 | 18.7685 μs | 未提供 | 用户报告 |
| 同上 | 放弃 | UB 流量减少但耗时增加 | 实验 A 行内融合 | 未提供本轮精度明细 | 19.3954 μs | 未提供 | 不在 B 中叠加 A；不将根因推断当作实测 |
| 沿用相同 case，配置待内网核对 | 待实测 | host 预算与 A5 实际对象不符 | 实验 B A5 完整 UB 预算 / T=1 单输出缓冲 | 待设备验证 | 未测 | 未测 | 仅实现候选，目标为相同口径下快于 main 且超过波动 |

采用开发参考 CHUNK_DEPENDENT/V1，来源 commit `a14961e9dfe8c8264768d67fb447101873da4ecf`。
仅沿用递推状态和 queue 生命周期的核对方法，无 Cube、跨核 workspace 或 chunk ping/pong。

## 静态验证与追溯

| 设计项 | 实现与验证 | 状态 |
| --- | --- | --- |
| 平台查询与 A5 标记 | CompileInfo → processor context；默认 false | 已编码；平台检测沿用仓内 GetSocVersion/ASCEND950 写法，CANN 编译待内网 |
| A5 所有 queue 与 tmp scratch 核算 | `op_host/recurrent_gated_delta_rule_a5_ub.h`，对照 InitLocalBuffers 对象逐项计数 | 已编码；含始终分配的 gamaInUb、每 queue slot 32B 对齐及 128B 余量 |
| profile 与容量边界 | `op_host/tests/a5_ub_profile_test.cpp` 的独立地址账本和策略检查 | 已编码；本地无 C++ 编译器，未执行 C++ 测试；另以 Python 静态资源模型枚举 40960 组条件通过 |
| T=1 的浅缓冲与 T>1 的候选 | host 选择，不读取设备 metadata | 静态分支检查通过，保留 main 的平衡 V 切片规则 |
| A2/A3 与 fast example | 非 A5 分支保持原逻辑；fast example 使用通用 kernel，默认 false | 从 processor 中移除新增 A5 入口后与 main 逐字一致 |
| 计算、ABI、同步 | 两个 kernel header、kernel 入口、tiling struct 与 main 比较 | 逐字一致，未携带实验 A；ACLNN/schema 未修改 |
| 文档、diff、测试资产 | 链接、语法、JSON、diff --check | 检查通过；原三份 JSON 与 main 相同，测试 Python AST 可解析，新增 C++ 括号与文档链接检查通过；不等同于编译 |

静态资源枚举覆盖 Nv=1..256、stateBytes=2/4、四种内部 gate 组合、T=1/2/8/16，
分别核对示例容量 192/248/256 KiB 和最小可行容量的前一字节/恰好容纳边界。
这些容量仅用于静态边界检查，实际实现始终使用平台查询值。Python 模型不是 C++ 执行或
CANN 编译结果；不得将其记录为已在 A5 跑过的用例。

静态容量例（T=1/g-only）中，R=128 的命名对象加余量为：Nv=1/BF16 150592 B，
Nv=1/FP32 216128 B，Nv=32/FP32 219520 B。main 的 kernel 和公开 ABI 无代码差异。

## 内网验证顺序

先构建并覆盖当前环境 wheel 的内嵌 OPP，再核对实际加载产物并重启 Python 进程：

```bash
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
./build_out/fla_npu_linux-*.run --install
python scripts/verify_libcust_opapi_md5.py --run-package <本轮实际run包>
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -scope=accuracy
```

新分支必须记录实际 TilingKey、vStep、dS/dO、tmpBytes 与 UB 查询值，先核对 out 和完整
stateRef 的精度，再跑确定性、mssanitizer 和同口径性能。TilingKey 仍只有 BF16/FP32 两类，
需覆盖每类的 T=1/T>1、g/gk/both、1/2 V 片，以及同核多 work、深缓冲 slot 回绕和片末排空。
必要时按开发流程启用独立执行记录，正式性能关闭记录。

正式用例包按 `tests/atk/README.md` 补齐并跑 all；原 main 的 51/88/157/188 条精度 JSON
含重复 state index 且与生成器不同，原 mss 仅有 BF16 最小例，需先协调这些既有资产问题。
本轮不修改 CPU 标杆、值域、容差或原始 perf case。设备验收完成前保留本开发记录。

A/B 对比保留相同 SoC/CANN、时钟状态、输入、预热/采样、重放方式和 kernel 范围。
仓库正式性能口径为 msprof op_summary 的 Task Duration(us)，调用链总时延单独记录。
对当前 BF16 最小例预期主要检查 profile 是否由 (2,2)→(1,1)；若 main 已为一片，
不预期由减少分片获得收益。FP32/Nv=32 的一片容量例只作为边界核对，不能冒充用户模型数据。
