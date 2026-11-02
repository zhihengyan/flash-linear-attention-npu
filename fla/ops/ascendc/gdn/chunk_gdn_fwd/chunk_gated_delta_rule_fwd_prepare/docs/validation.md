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

每轮先构建并安装单算子 run 包，确认 MD5 一致后重启测试 Python 进程：

```bash
bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu --ops=chunk_gated_delta_rule_fwd_prepare -j8

./build_out/fla_npu_linux-*.run --install

python scripts/verify_libcust_opapi_md5.py --run-package <本轮 run 包>
```

## 性能追踪

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 推理 case 0 | 基线 | Prepare Stage3/7 | 原始实现 | 用户给定稳定基线 | 1705.298 µs；范围 1703.162～1710.482 µs | 待本轮 profiling | 三轮波动 7.321 µs |
| 训练 case 1 | 基线 | Prepare Stage3/7 | 原始实现 | 用户给定稳定基线 | 2707.735 µs；范围 2704.416～2709.092 µs | 待本轮 profiling | 三轮波动 4.676 µs |
| 两条 case | Stage3 / 待设备实测 | 64×64 仅 2016 个严格下三角元素，原实现对 4096 元素 Mul/Store | 仅将 Stage3 Mul/Store 改为 lower mask；Load、打包、VCS、同步不变 | 待 500 条 | 待至少两轮 | 待核对 Stage3 Vector 时间 | 精度通过且端到端收益超过噪声才采用 |
| 两条 case | Stage7 / 待设计实测 | W/U 重复 LoadA，K=V=128 存在两个 N=128 MMAD | 仅 K=V=128 合并 packed N=256；V=256 原路径 | 未实施、未测试 | 未测试 | 未测试 | Stage3 闭环后独立开展 |

## 追溯与检查

| 设计项 | 代码位置与实际实现 | 当前证据 / 状态 |
| --- | --- | --- |
| Stage3 严格下三角 | `chunk_gated_delta_rule_fwd_prepare_vf.h` `MulKktGateVF` | 已实现 masked Mul/Store；待构建与设备验证 |
| Stage3 上三角零值 | 同文件 `GateLowerLVF` | 原实现全行 Select 后 Store，零值先于 masked Store 建立 |
| Stage7 packed W/U | `chunk_gated_delta_rule_fwd_prepare.h`、`_matmul.h` | 仅方案，未修改代码 |
| 资源与同步 | Prepare `InitOnChip`、Stage3/7 | Stage3 未改变地址、容量、event 或计数 |

本地无生产 NPU；构建、500 条精度和性能均不得标记为已执行。若 Stage3 编译或精度失败，
优先回退本项并检查 masked Store 语义，不以修改 CPU 标杆、容差或上三角数学定义规避。
