# Prepare 开发期验证记录

## 固定验收口径

- 代码分支：`feat/gdn-opt`，本轮基线 `ea1f61a1`；当前已推送提交 `11c963e8`，其中算子代码与 `a5ba454e` 一致，新增六条小 case 性能用例。
- 当前验证：`c15adc11` 完整回退 Stage7 packed MMAD 后，正式融合精度已通过，仅保留 fwd_h 和小 case 短 pack 并行优化。`a5ba454e` 将 fwd_h 的 VF 列宽改为调用前分派和编译期常量，Prepare 实现不再改动。无干扰 A/B 中，正式 case 0/1 基本持平；小 case 2～6 降低 7.60%～15.62%，case 7 增加 1.99%，仍需独立复测确认波动。
- 硬件与设备：Ascend 950，device 2。
- 精度：正式 500 条双标杆矩阵，每 25 条自动分片；最大相对误差比例 5、平均相对误差比例 1.5、均方根误差比例 1.5，沿用 ATK 判定。
- 性能：固定两条正式 case，每项至少两轮，与正式中位数比较；小 case 使用同一 JSON 做基线/优化 A/B。不以 ACL event、理论核利用率或单 Stage timer 替代端到端结论。

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 2

PERFORMANCE_START=0 PERFORMANCE_END=1 ATK_OUTPUT_ROOT="tests/atk/chunk_gated_delta_rule_fwd/atk_output/perf_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)" bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=2 -scope=performance --soc=ascend950

PERFORMANCE_START=2 PERFORMANCE_END=7 ATK_OUTPUT_ROOT="tests/atk/chunk_gated_delta_rule_fwd/atk_output/small_perf_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)" bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=2 -scope=performance --soc=ascend950
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
| 小 case | 短 pack / 一轮无干扰 A/B | 原模型只看最忙核 task 数，case 1 会从 pack=4 错选 pack=1，case 0 平局仍选 4 | 仅 fullPacks<numCore 时评估短 pack，平局保留大 pack | Prepare slim 通过；Stage7 回退后正式融合精度通过 | case 4：52.88→47.70 µs | 仅 Prepare 路径耗时降低 9.80% | 两条正式 case 均恢复 pack=4；小 case 仍需独立复测 |

### 小 case A/B 矩阵

完整 shape 见 `tests/atk/chunk_gated_delta_rule_fwd/README.md`。预期路径按 28 个 AIC 静态计算，
尚不代表已观察到的运行时选择。以下为一轮无干扰端到端数据，所有 case 仍需独立复测。

| case id | 预期 pack / valueSplit | 归因目的 | ea1f61a1 基线 | 优化版本 | 耗时变化 |
| --- | --- | --- | ---: | ---: | ---: |
| 2 | 1 / 2 | G=1，两项联合 | 58.04 µs | 53.63 µs | **-7.60%** |
| 3 | 2 / 2 | G=2，两项联合 | 57.99 µs | 48.93 µs | **-15.62%** |
| 4 | 2 / 1 | 仅 Prepare | 52.88 µs | 47.70 µs | **-9.80%** |
| 5 | 4 / 2 | 仅 fwd_h | 114.80 µs | 101.63 µs | **-11.47%** |
| 6 | 2 / 2 | varlen 联合 | 56.00 µs | 48.00 µs | **-14.29%** |
| 7 | 4 / 1 | 未触发对照 | 82.50 µs | 84.14 µs | **+1.99%**，待复测确认是否回退 |

## 待验证矩阵

1. **构建**：对 Stage7 回原双 MMAD、保留短 pack 门控和 fwd_h 修复的版本重新构建九算子 wheel。
2. **正式精度**：重跑完整 500 条双标杆，确认 12/13/14、453/380/435 等历史失败用例及全部其他
   用例均通过；未全部通过前不运行性能结论。
3. **短 pack 调度**：覆盖 G=1/2 的 pack=1/2/4 可达组合、最后一个不完整 pack、跨 pack L1
   槽复用和最终 W/U drain；G=3/4 应保持原调度。
4. **性能**：正式精度全部通过后，两条正式 case 至少各复测两轮并保留原始值。
5. **小 case 性能证据**：运行 case 2～7，记录实际 pack 选择、理论最忙核 task 数和端到端时间；
   case 4 隔离短 pack，case 7 检查未触发路径是否保持基线。

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
