# Prepare 开发期验证记录

## 固定验收口径

- 代码分支：`feat/gdn-opt`，本轮基线 `ea1f61a1`；当前提交 `ec611538`。
- 当前验证：Prepare slim 与 `ec611538` 的 fwd_h 单算子精度均通过；当前融合矩阵仍有 435、380、277、279、280 等上一轮失败用例，完整集合待汇总。组合版本性能为 7783/10765 µs，严重低于 1705.298/2707.735 µs 基线。短 pack 原成本模型会在大任务中因极小 task 差选择 pack=1，本地已限制为 fullPacks<numCore 才评估；fwd_h 同步增加 half-head 适用门控，待重新构建和复测。
- 硬件与设备：Ascend 950，device 2。
- 精度：正式 500 条双标杆矩阵，每 25 条自动分片；最大相对误差比例 5、平均相对误差比例 1.5、均方根误差比例 1.5，沿用 ATK 判定。
- 性能：固定两条正式 case，每项至少两轮，与正式中位数比较；不以 ACL event、理论核利用率或单 Stage timer 替代端到端结论。

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 2

ATK_OUTPUT_ROOT="tests/atk/chunk_gated_delta_rule_fwd/atk_output/perf_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)" bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=2 -scope=performance --soc=ascend950
```

每轮先构建并安装包含 DUT 及其公开入口依赖的 9 算子 wheel，确认 MD5 一致后重启测试
Python 进程。不得用只含 Prepare 的单算子包覆盖精度、性能共用环境，否则 `libcust_opapi.so`
会缺少公开融合算子和 `chunk_local_cumsum` 等符号；MD5 一致不代表符号齐全。

```bash
FLA_NPU_SOC=ascend950 \
FLA_NPU_OPS=chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_fwd_prepare,chunk_local_cumsum,chunk_scaled_dot_kkt,solve_tri,recompute_w_u_fwd,chunk_gated_delta_rule_fwd_h,chunk_fwd_h,chunk_fwd_o \
python3 scripts/build_wheel.py

WHEEL=$(ls -t dist/*.whl | head -n 1)
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL"

python scripts/verify_libcust_opapi_md5.py
```

## 性能追踪

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 推理 case 0 | 基线 | Prepare Stage3/7 | 原始实现 | 用户给定稳定基线 | 1705.298 µs；范围 1703.162～1710.482 µs | 待本轮 profiling | 三轮波动 7.321 µs |
| 训练 case 1 | 基线 | Prepare Stage3/7 | 原始实现 | 用户给定稳定基线 | 2707.735 µs；范围 2704.416～2709.092 µs | 待本轮 profiling | 三轮波动 4.676 µs |
| 两条 case | Stage3 / 负收益，放弃并回退 | 原实现对 4096 个元素执行 Mul/Store；候选只减少有效 lane | 候选仅将 Mul/Store 改为 lower mask，Load、打包、VCS、同步不变 | 正式 500 条未完整跑通 | 用户报告两次均值约 1710／2715 µs，基线为 1705.298／2707.735 µs | 无有效收益 | 循环、Load、Mul 和 Store 调用次数均未减少，却新增 UpdateMask 和 masked Store 开销；训练慢约 7.265 µs，超过基线波动，不能保留 |
| 两条 case | Stage7 packed W/U / 独立 slim 通过，待融合回归 | 原路径重复 LoadA，K=V=128 有两个 N=128 MMAD | 仅 K=V=128 合并 packed N=256；V=256 原路径 | Prepare slim 通过；当前融合矩阵仍有失败 | 组合版本 7783/10765 µs，不能作为单候选独立值 | 待逐 kernel profiling | 构建与独立精度通过；待并行门控修复后复测，不预写收益 |
| 小 case | 短 pack / 独立 slim 通过，增加大任务门控后待复测 | 原模型只看最忙核 task 数，case 1 会从 pack=4 错选 pack=1，case 0 平局仍选 4 | 仅 fullPacks<numCore 时评估短 pack，平局保留大 pack | Prepare slim 通过；当前融合矩阵仍有失败 | 组合版本 7783/10765 µs，不能作为单候选独立值 | pack 固定开销及流水粒度可能增加 | 两条正式 case 均恢复 pack=4；后续只选择存在空闲核的 shape |

## 待验证矩阵

1. **构建与模板实例化**：`WuPackedMatmulToUb` 的 TLA `GetTile`、packed `TileMmadTla`、
   `CopyL0CToDst` 已在目标 CANN 构建通过。
2. **Stage7 正常与尾部**：覆盖 `K=V=128` 的完整 64 行 chunk、`M<64` 尾 chunk、G=1/2/3/4，
   检查 W/U 两个 N=128 半区没有互换、串列或越界；覆盖 V=256 原路径回归。
3. **短 pack 调度**：覆盖 G=1/2 的 pack=1/2/4 可达组合、最后一个不完整 pack、跨 pack L1
   槽复用和最终 W/U drain；G=3/4 应保持原调度。
4. **精度隔离**：先确认 `chunk_fwd_h` 单算子结果，并运行 Prepare 独立 slim 精度集；必要时每次只
   回退 packed W/U、短 pack 或 half-head 中的一项，使用相同失败 case 复现。正式 500 条全部通过后
   才运行两条正式性能 case。每个性能结果至少复测两轮并保留原始值。
5. **小 case 性能证据**：额外选择短 pack 实际降低最忙核 task 数的 shape，同时记录 pack 选择、
   理论最忙核 task 数和端到端时间。512 token 只有 8 个序列 chunk，但 Prepare task 数是
   `8*HV`；若所有核已各有一个 pack，短 pack 不会自然产生收益。

## 追溯与检查

| 设计项 | 代码位置与实际实现 | 当前证据 / 状态 |
| --- | --- | --- |
| Stage3 严格下三角 | `chunk_gated_delta_rule_fwd_prepare_vf.h` `MulKktGateVF` | 已回退为原始全量 Mul／Store；负收益 masked 候选不再保留 |
| Stage3 上三角零值 | 同文件 `GateLowerLVF`、`MulKktGateVF` | GateLowerLVF 先写零，原始 KKT⊙G 全行乘法和写回保持对角线、上三角为零 |
| Stage7 packed W/U | `chunk_gated_delta_rule_fwd_prepare.h`、`_matmul.h` | shared packed-NZ L0B、一次 N=256 MMAD、两个 L0C 半区取出已编译；正式矩阵存在精度失败，待隔离 |
| Stage7 尾部与同步 | `Stage7_AicOne`、`WuPackedMatmulToUb` | 静态确认只写 `chunk.M` 行，event 0 保护 L0 输入和 L0C 生命周期；设备精度未通过 |
| 短 pack | `PrepareState::TasksPerPack`、AIC/AIV pack 循环 | 静态确认按实际 `cur.nThis` 执行且不拆 HK 共享组；正式矩阵存在精度失败，待隔离 |
| 资源 | Prepare `InitOnChip`、Stage7 | packed 使用 32 KiB L0B 与 64 KiB L0C，位于现有 S7 bank，不新增 workspace |

本地无生产 NPU。Stage3 的构建和 MD5 检查曾由用户确认通过，但正式 500 条精度未完整跑通，
性能也只有用户报告的两次均值，没有每轮原始值，因此不得标记为精度通过或稳定收益。本轮
Stage7、短 pack 已构建，但正式 500 条只有 450 条精度通过，必须先定位并修复失败用例，再至少进行
两轮同口径性能复测。
