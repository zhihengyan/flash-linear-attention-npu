# ChunkFwdH 开发期验证记录

## 当前状态

- 分支与基线：`feat/gdn-opt`，基线 `ea1f61a1`；当前已推送提交 `c15adc11`，本地另有 VF 编译期分派修复待提交。
- 实现范围：仅在 Ascend 950、scalar-g、FP32 state、`state_v_first=false`，原任务少于物理 AIC 核且切分后 blockDim 增加时启用 `valueSplit=2`。
- 当前证据：`c15adc11` 完整回退 packed W/U 后，正式融合精度已通过；但 case 0 性能为 3518 µs，仍为 1705.298 µs 基线的约 2.06 倍。该 case 已恢复 valueSplit=1、pack=4，因此本地继续修复未切分路径中的动态列宽热循环：Stage1、Stage3、S-1 改为 VF 调用前分派和模板常量列宽。修改后的构建、精度和性能均待上机验证。
- 不得将任务数增加、理论核利用率或静态检查写成精度通过或性能收益。

## 固定验收环境

硬件为 Ascend 950，device 2。先构建并安装包含公开融合入口及依赖的 9 算子 wheel，再启动
新的测试 Python 进程。正式端到端基线为：推理 case 0 中位数 1705.298 µs，三轮范围
1703.162～1710.482 µs；训练 case 1 中位数 2707.735 µs，三轮范围 2704.416～2709.092 µs。

```bash
FLA_NPU_SOC=ascend950 \
FLA_NPU_OPS=chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_fwd_prepare,chunk_local_cumsum,chunk_scaled_dot_kkt,solve_tri,recompute_w_u_fwd,chunk_gated_delta_rule_fwd_h,chunk_fwd_h,chunk_fwd_o \
python3 scripts/build_wheel.py

WHEEL=$(ls -t dist/*.whl | head -n 1)
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL"

python scripts/verify_libcust_opapi_md5.py

bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=2 -scope=accuracy --soc=ascend950

bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 2

ATK_OUTPUT_ROOT="tests/atk/chunk_gated_delta_rule_fwd/atk_output/perf_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)" bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=2 -scope=performance --soc=ascend950
```

验证顺序固定为 `chunk_fwd_h` 单算子精度 → 正式 500 条双标杆矩阵 → 两条正式性能 case。
性能 case 至少复测两轮并保存逐轮原始值，与上述中位数及波动范围比较。

## 性能追踪

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 推理 case 0 | half-head / 严重负收益，增加适用门控后待复测 | 原 32 个 head task 已超过 28 核仍切为 64 个 half-head | 仅当原任务少于物理核且 split 后 blockDim 增加时启用 | fwd_h 单算子通过；融合矩阵仍有失败 | 1705.298 → 7783 µs（组合版本） | 待逐 kernel profiling | 固定开销及 W/K 重复装载增加；门控后正式 case 恢复 valueSplit=1 |
| 训练 case 1 | half-head / 严重负收益，增加适用门控后待复测 | 原 64 个 head task 已超过 28 核仍切为 128 个 half-head | 同上 | 同上 | 2707.735 → 10765 µs（组合版本） | 待逐 kernel profiling | 门控后正式 case 恢复 valueSplit=1；组合值不作为单候选独立收益 |
| 推理 case 0 | 编译期 VF 列宽 / 本地实现，待上机 | packed 回退且 valueSplit=1 后仍退化，运行时 valueSize 留在热循环 | VF 调用前一次分派；模板常量确定 64/128 列循环、stride 和寄存器分支 | c15adc11 正式融合精度通过；本修改待回归 | 1705.298 → 3518 µs（c15adc11）；本修改待测 | 目标是恢复原 128 列常量展开与地址优化 | half-head 保留；不能将待测修改写成收益 |

## 必测覆盖

| 类别 | 必测内容 | 重点检查 |
| --- | --- | --- |
| 模板选择 | A5 scalar-g + FP32 state + `state_v_first=false`，且原任务少于物理核、split 后 blockDim 增加时选择 `valueSplit=2`；其余路径选择 1 | 未改动路径保持原行为，host/kernel tiling 字段一致 |
| 调度 | `N*HV` 小于、等于和大于物理核数；连续区间跨 sequence；奇偶 half-head；最后一个核不满 | AIC/AIV 解码一致，每个 `(sequence,hv,vPart)` 恰好执行一次，无游标跳过 |
| 初态与零态 | 无 initial state、有 FP32 initial state；S-1 H0 转换和首 chunk ZERO_STATE | 两个 64 列半区不串扰，不错误读取第二组向量寄存器 |
| 状态输出 | 无 final state、有 FP32 final state；最后一个 chunk 的 Stage2/Stage3 裁剪 | final state 两个半列均写回，右半列基址和行 stride 正确 |
| 序列边界 | 单 chunk、两个 chunk、多 chunk、`T%64!=0` 尾 chunk；dense 与 varlen | H 的 chunk 前缀、tail 有效行、W/K lookahead 和末块写回正确 |
| GM/片上布局 | H/U/V_new/initial/final state 的左右半列；right/state workspace | GM 行 stride 保持 128，片上紧凑 64 列，无跨行读写或槽重叠 |
| 流水协议 | 单 head pipeline、空闲 AIV1、连续 work unit | P/D/right ready/free、ROUND_DONE/ACK 无死锁，state 不跨任务残留 |
| 性能 | 两条正式 case，并补充一个原本存在明显核间尾波的小 case | 同时记录 blockDim、每核 half-head 数和端到端时间，识别重复 W/K 装载代价 |

## 静态复核记录

- Host 以 `N*HV*valueSplit` 计算任务数和 blockDim；AIC/AIV 均通过 `FwdHTotalHeadTasks` 与
  `FwdHResolveCoreHeadRange` 取得相同区间。
- `valueSplit=2` 时 `unitHeads=1`，`cursor+=1`；`hv=taskInSequence/2`，
  `vBegin=(taskInSequence&1)*64`，两侧一致。
- H/U/V_new/state 的 GM 基址包含 `vBegin`，半列 copy 的源/目的 gap 均为 64 列；right 和
  非驻留 state 的 workspace 两侧均使用 `head.vSize` 计算 slot 元素数。
- `LoadStage0H` 的 half-head GM tensor 使用带 `LayoutH` 标签的完整 `[K,128]` layout，再取
  `[K,64]` tile；GM 行 stride 保持 128，同时命中既有 `TileCopyTla` 特化。通用动态 layout
  导致的 GM→L1 拷贝编译错误已修复，并已通过上机构建。
- Stage1、Stage3、S-1 以 `valueSize == FWD_H_V / 2` 明确选择 64 列单 FP32 寄存器路径，
  128 列走原双寄存器实现；`ec611538` 已修正此前使用 `BF16_PER_REG` 导致的分支选反，fwd_h
  单算子精度全部通过，正式融合矩阵待重跑。
- V 维切分不需要跨核归约；K/raw-K 保持完整。每个 half-head 的 FP32 rolling state 独立于
  另一 half-head，原单 head ready/free 与 round 握手保持不变。

以上均为代码审查结论，不替代构建和设备验收。
