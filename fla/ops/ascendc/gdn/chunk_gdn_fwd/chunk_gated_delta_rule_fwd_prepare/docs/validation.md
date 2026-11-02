# Prepare 开发期验证记录

## 固定验收口径

- 代码基线：`feat/gdn-opt`，起始 HEAD `2d5a8f24`。
- 硬件与设备：Ascend 950，device 2。
- 精度：正式 500 条双标杆矩阵，每 25 条自动分片；最大相对误差比例 5、平均相对误差比例 1.5、均方根误差比例 1.5，沿用 ATK 判定。
- 性能：固定两条正式 case，每项至少两轮，与正式中位数比较；不以 ACL event 或单 Stage timer 替代端到端结论。

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
| 两条 case | Stage7 / 待设计实测 | W/U 重复 LoadA，K=V=128 存在两个 N=128 MMAD | 仅 K=V=128 合并 packed N=256；V=256 原路径 | 未实施、未测试 | 未测试 | 未测试 | Stage3 回退闭环后独立开展 |

## 追溯与检查

| 设计项 | 代码位置与实际实现 | 当前证据 / 状态 |
| --- | --- | --- |
| Stage3 严格下三角 | `chunk_gated_delta_rule_fwd_prepare_vf.h` `MulKktGateVF` | 已回退为原始全量 Mul／Store；负收益 masked 候选不再保留 |
| Stage3 上三角零值 | 同文件 `GateLowerLVF`、`MulKktGateVF` | GateLowerLVF 先写零，原始 KKT⊙G 全行乘法和写回保持对角线、上三角为零 |
| Stage7 packed W/U | `chunk_gated_delta_rule_fwd_prepare.h`、`_matmul.h` | 仅方案，未修改代码 |
| 资源与同步 | Prepare `InitOnChip`、Stage3/7 | Stage3 回退不改变地址、容量、event 或计数 |

本地无生产 NPU。Stage3 的构建和 MD5 检查曾由用户确认通过，但正式 500 条精度未完整跑通，
性能也只有用户报告的两次均值，没有每轮原始值，因此不得标记为精度通过或稳定收益。后续每项
优化仍须先完成正式精度，再至少进行两轮同口径性能复测；每轮只改变一个主要变量。
