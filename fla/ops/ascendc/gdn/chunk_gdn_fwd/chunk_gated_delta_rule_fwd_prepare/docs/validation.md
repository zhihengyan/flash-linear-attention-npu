# Prepare 开发期验证记录

## 固定验收口径

- 代码分支：`feat/gdn-opt`，本轮基线 `ea1f61a1`；当前已推送提交 `c15adc11`。
- 当前验证：`c15adc11` 完整回退 Stage7 packed MMAD 后，正式融合精度已通过，仅保留 fwd_h 和小 case 短 pack 并行优化。case 0 性能仍为 3518 µs，高于 1705.298 µs 基线；该 case 的短 pack 已恢复 4、valueSplit 已恢复 1。当前本地修改仅将 fwd_h 的 VF 列宽改为调用前分派和编译期常量，Prepare 不再改动；性能待上机复测。
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
| 两条 case | Stage7 packed W/U / 精度失败，放弃并完整回退 | 原路径重复 LoadA，K=V=128 有两个 N=128 MMAD | 候选合并 packed N=256；当前恢复原 W/U 双 MMAD | Prepare slim 曾通过，但正式融合矩阵仍有 12/13/14、453/380/435 等失败 | 组合版本 7783/10765 µs，不作为单候选独立值 | 不再保留 packed 路径 | 用户决定完整回退；新增 TLA include、helper 和调用分支均删除 |
| 小 case | 短 pack / 独立 slim 通过，增加大任务门控后待复测 | 原模型只看最忙核 task 数，case 1 会从 pack=4 错选 pack=1，case 0 平局仍选 4 | 仅 fullPacks<numCore 时评估短 pack，平局保留大 pack | Prepare slim 通过；Stage7 回退后待正式矩阵复测 | 组合版本 7783/10765 µs，不能作为单候选独立值 | pack 固定开销及流水粒度可能增加 | 两条正式 case 均恢复 pack=4；后续只选择存在空闲核的 shape |

## 待验证矩阵

1. **构建**：对 Stage7 回原双 MMAD、保留短 pack 门控和 fwd_h 修复的版本重新构建九算子 wheel。
2. **正式精度**：重跑完整 500 条双标杆，确认 12/13/14、453/380/435 等历史失败用例及全部其他
   用例均通过；未全部通过前不运行性能结论。
3. **短 pack 调度**：覆盖 G=1/2 的 pack=1/2/4 可达组合、最后一个不完整 pack、跨 pack L1
   槽复用和最终 W/U drain；G=3/4 应保持原调度。
4. **性能**：正式精度全部通过后，两条正式 case 至少各复测两轮并保留原始值。
5. **小 case 性能证据**：额外选择短 pack 实际降低最忙核 task 数的 shape，同时记录 pack 选择、
   理论最忙核 task 数和端到端时间。

## 追溯与检查

| 设计项 | 代码位置与实际实现 | 当前证据 / 状态 |
| --- | --- | --- |
| Stage3 严格下三角 | `chunk_gated_delta_rule_fwd_prepare_vf.h` `MulKktGateVF` | 已回退为原始全量 Mul／Store；负收益 masked 候选不再保留 |
| Stage3 上三角零值 | 同文件 `GateLowerLVF`、`MulKktGateVF` | GateLowerLVF 先写零，原始 KKT⊙G 全行乘法和写回保持对角线、上三角为零 |
| Stage7 W/U | `chunk_gated_delta_rule_fwd_prepare.h`、`_matmul.h` | packed 候选已完整回退；恢复原 W/U 双 MMAD 和既有 Fixpipe/drain 路径 |
| Stage7 尾部与同步 | `Stage7_AicOne` | 恢复原 event 0/1 生命周期和有效行写回；待上机回归 |
| 短 pack | `PrepareState::TasksPerPack`、AIC/AIV pack 循环 | 仅 fullPacks<numCore 时缩短；不拆 HK 共享组，待正式矩阵复测 |
| 资源 | Prepare `InitOnChip`、Stage7 | 恢复原 L0/L1/UB 使用，不保留 packed 资源或新增 workspace |

本地无生产 NPU。历史候选的 slim 或静态结果不能替代正式 500 条验收。当前版本必须重新构建，
完成正式精度后再至少进行两轮同口径性能复测。
