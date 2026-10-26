# A5 实验 A 开发记录

## 范围与基线

- main 基线：`571d7acf`；开发分支：`codex/gdn-a5-vf-row-fusion`。
- 用户指定参考：2026-10-23 的 `gdn-a5-decode-analysis.md` 实验 A；不引入实验 B。
- 2026-10-26 用户明确验证环境在内网，本轮仅编码和静态检查。当前 Windows 会话无
  CANN/NPU/torch/ATK，因此未运行设备基线、CANN 编译或 NPU 验收，不声明性能收益。
- 当前算子此前没有 api/design/validation 三份阶段文档；本轮补接口索引和受影响 Stage 设计。
  不将既有 AIV_ONLY 调度描述成新的 AIC/AIV 分核方案。
- 开发参考角色 `CHUNK_DEPENDENT`，版本 V1，来源 commit
  `a14961e9dfe8c8264768d67fb447101873da4ecf`。采用递推状态生命周期和逐目标验证原则；
  本算子无 Cube、跨核 workspace 或 chunk ping/pong，不复制参考算子的架构或容量常量。
- 沿用当前 CPU 标杆、输入生成器及 `mixed_tolerance_bm`；不改值域和容差。

## 实验追踪

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 未提供模型 case，无量化性能目标 | 基线 | 源码有两遍 state 读写、dot 落 UB | 保持原 tiling 和 queue | 未运行 NPU | 未测 | 未测 | 仅源码基线 |
| 同上 | 待实测 | 重复 VF UB 访问 | 逐行融合；仅 L=1 省 FP32 scratch 写入 | 待内网 A5 验证 | 未测 | 未测 | 本轮仅编码和静态检查，不作采用收益结论 |

## 实现追溯与静态检查

| 设计项 | 实现 / 检查位置 | 状态 |
| --- | --- | --- |
| 每行 state 驻留、dot 首 lane 广播 | arch35 header 的 `ComputeRecurrentGatedDeltaRuleVF` | 已编码；两个 row 循环合为一个，dot 不再落 UB |
| 单 token 不写递推 scratch，多 token 保留 FP32 | VF 的 saveState 与 `ProcessHead` 的实际 L 分支 | 已编码；scratch stores 全部位于 if constexpr(saveState) 内 |
| 所有 gate 分支透传模板参数 | `Compute<saveState>` 的四个 gate 实例 | 静态核对通过；实际分支执行待设备确认 |
| 输出打包前 barrier、BF16/FP32 layout/cast | VF 尾部及 diff 复核 | 最后的 barrier 和 out0 起至 VF 末尾均与基线一致 |
| TilingKey、队列深度、UB 地址、GM 写回 | host/kernel 调用 diff | host 未修改；分配、搬运、释放函数逐函数对照一致 |
| 既有 ATK 用例可解析、覆盖新分支 | generator 与三份 JSON 的静态交叉核对 | 三份 JSON 可解析，两个 L 分支的六种 dtype/gate 组合均有既有用例 |
| 生成结果一致 | 现有精度 JSON 与 generator | 发现 main 原有 4 条不一致，见下表；本轮不修改测试资产 |
| diff 空白、文档链接、工作树范围 | git diff --check、Python 本地链接检查和 Git status | 检查通过；仅 A5 header 和三份阶段文档 |

静态检查使用本地 Python 标准库：检查 C++ 括号配对、四个 gate 模板的参数透传、scratch
写入条件、实际 L 分支及与 main 的函数级差异。Python AST 检查现有测试文件，导入 generator
生成 200 条内部校验通过的候选，用 JSON 解析当前三份用例文件。上述结果不等同于 CANN 编译。

现有测试资产问题已用 `git show main:<文件>` 确认在 main 中相同：

| 精度 case id | 已提交 JSON 的 state_indices | generator 的 state_indices | 影响 |
| --- | --- | --- | --- |
| 51 | `[0,0,0]`，block_num=1 | `[1,2,0]`，block_num=3 | 违反接口索引唯一约束 |
| 88 | `[8,7,6,5,0,2,4,6,8]` | `[8,7,6,5,0,2,4,1,3]` | 违反接口索引唯一约束 |
| 157 | `[0,0]`，block_num=1 | `[1,0]`，block_num=2 | 违反接口索引唯一约束 |
| 188 | `[0,0,1,1,2,2,3,3]`，block_num=4 | `[0,2,4,6,1,3,5,7]`，block_num=8 | 违反接口索引唯一约束 |

后续正式验收前需先按测试流程协调这四条用例和生成器；本轮保留原文件，避免将测试基线修订
混入实验 A 的 kernel diff。不能将“200 条 JSON 可解析”记录为“200 条合法用例验收通过”。

## 内网 A5 后续验证

本节是后续设备验证计划，不是本轮已执行结果。按 `04` 构建/安装并确认产物后再测：

```bash
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
./build_out/fla_npu_linux-*.run --install
python scripts/verify_libcust_opapi_md5.py --run-package <本轮实际run包>
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -scope=accuracy
```

重启测试 Python 进程以加载当前产物。精度先验证 out 与完整原地 stateRef，再执行确定性、
mssanitizer 与性能；正式验收按 `tests/atk/README.md` 补齐种子、三类映射和用例包后跑 `all`。
现有 perf/mss 文件只含 BF16 g-only 最小例，不能代表完整模型性能或两种 dtype 的内存覆盖。

既有精度矩阵的代表性覆盖（均保持现有输入值域和种子）：

| 分支 | BF16 state case id | FP32 state case id |
| --- | --- | --- |
| L=1 / g | 0 | 1 |
| L=1 / gk | 174 | 83 |
| L=1 / both | 95 | 132 |
| L>1 / g | 2 | 15 |
| L>1 / gk | 3 | 61 |
| L>1 / both | 4 | 9 |

解决上述既有用例问题后回归完整矩阵，尤其 prefix/零长度、混合 L、accepted token、GVA、state 非连续、
重复 decode。`L==1` 和 `L>1` 的实际分支选择需用设备记录确认，静态 shape 不能作为已执行证据。
以现有 CPU 标杆 FP32 递推及逐 token state 快照为预期；必要时按精度定位文档采集 VF
的 stateOut/attnOut，中间 FP32 scratch 仅在 saveState=true 时具有后继消费者。

性能使用用户真实模型 case，在同一 A5 软件/硬件环境、输入、预热和采样条件下比较 main 与
本分支的 msprof `op_summary Task Duration(us)`，同时检查寄存器 spill 和 UB/VF 指标。
未指定真实 case 和测量配置前，不以现有最小 perf 用例宣称模型提速。设备验收完成后再按 `05`
归档到 ATK README 并删除本开发记录；本轮保留待验证状态。
