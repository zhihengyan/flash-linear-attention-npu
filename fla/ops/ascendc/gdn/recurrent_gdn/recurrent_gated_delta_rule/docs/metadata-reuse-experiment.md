# A5 元数据复用实验

本轮只实验整数元数据复用。原 head 扫描、VF 数学、末 token scratch 省写、缓冲与同步保持。
先验证这一项，再决定是否实验缩小 head 遍历范围。

## 1. 实验版本和预期变化

| 版本 | 说明 |
|---|---|
| a6d33dbb | 对照基线，kernel 同 93f5c70a |
| c031ed24，及后续仅文档提交 | 当前 codex/gdn-a5-tail-pmu-profile 的元数据复用候选 |

ComputeAvgload 返回已读到的最后一个序列长度；仅 B=1 时 ProcessGeneral 复用它。
ProcessHead 在单 token 序列中复用已读的初始 state index，多 token 仍逐 token 读取输出索引。
accepted-token 的原校验保留：单 token 的合法 accepted 只能为 1，初始和输出索引位置因此一致。

对 case 1 的一个有效核，源码级访问预期如下：

| 元数据 | 基线 | 候选 |
|---|---:|---:|
| actualSeqLengths | 3 次：统计长度、prefix、再次读取长度 | 2 次：统计长度、prefix |
| ssmStateIndices，若 vStep=128 | 2 次：初始和输出 | 1 次：初始索引复用 |

这是源码访问次数，不是已经测得的机器指令数。编译器可能已合并旧访问，新增选择条件及寄存器
生存期也可能抵消收益。tensor 的 GM 读写和数学计算量没有减少，不预设一定加速。

本轮仅在上一轮 Ascend950DT_9582 环境比较 case 1，保持 B=T=1、Hk=16、Hv=32、BF16 state、g-only。
该环境已有一轮 BasicInfo 均值 6.472764 us、中位数 6.502314 us、CV 4.2365%，
设备对照以补采基线为准，不能仅与这一轮历史均值相减判定小幅收益；当前回传结果见第 5 节。

## 2. 离线控制检查

在仓库根目录执行：

```bash
python -B -m unittest discover -s tests/atk/recurrent_gated_delta_rule/scripts -p 'test_*.py' -v
```

测试检查 VF、原单 work、缓冲/搬运 helpers、head 分配、公开 ABI 和 A2/A3 源码保持；
同时检查元数据复用严格限定在设计条件中。
若 g++ 或 clang++ 可用，还会抽取本次及 a6d33dbb 的生产 Scalar 控制函数，编译对照其
任务、输入/输出地址和访问计数，覆盖空序列、非法长度/prefix、accepted、非零 index、padding stride、
多 V 分片、多个核数和输出队列深度。它不运行 Vector 数学，也不能代替 NPU 精度测试。
缺少编译器时该项会明确显示 skipped，不算控制流执行通过。

## 3. 分别构建、安装并验证设备精度

基线和候选分别构建并保留各自产物，每次比较前安装对应版本，使用新的 Python 进程。
本轮只修改私有 kernel，可以按仓库流程构建单算子 run 包：

```bash
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
```

按构建输出选择当前架构的 run 包，下面以 x86_64 为例，其他架构替换路径：

```bash
./build_out/fla_npu_linux-x86_64.run --install
python scripts/verify_libcust_opapi_md5.py --run-package build_out/fla_npu_linux-x86_64.run
```

先验证新增性能 case 1 的精度：

```bash
(
  cd tests/atk/recurrent_gated_delta_rule
  atk node --name npu_dut --backend npu --devices 0 \
    node --name cpu_golden --backend cpu \
    task -c atk_recurrent_gated_delta_rule_perf.json --task accuracy --bm_device cpu \
    -p executor_recurrent_gated_delta_rule.py -s 1 -e 2
)
```

再回归原 200 条精度用例，涵盖 B=1/多序列、空序列、accepted、多 token、BF16/FP32、gate 和 stride：

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=accuracy
```

内存、确定性及其覆盖边界仍按 [开发记录](validation.md) 和 [ATK README](../../../../../../../tests/atk/recurrent_gated_delta_rule/README.md) 执行。
没有设备结果前，不能把离线检查写成精度通过。

## 4. 同环境交错测量 BasicInfo

从仓库根目录执行。安装基线后用 LABEL=baseline，安装候选后用 LABEL=metadata_reuse。
每次更新 ROUND，输出新目录；例如完成至少三组基线/候选交错对照，各轮独立汇总。

```bash
LABEL=metadata_reuse
ROUND=1
SCRIPT="$PWD/tests/atk/recurrent_gated_delta_rule/scripts/msprof_op_rgdr.py"
mkdir -p tests/atk/recurrent_gated_delta_rule/runs/msprof_op
OUT=$(mktemp -d "$PWD/tests/atk/recurrent_gated_delta_rule/runs/msprof_op/${LABEL}_r${ROUND}_XXXXXXXX")
msprof op \
  --application="python $SCRIPT --repo-root $PWD --device 0 --case-id 1 --calls 20 --label $LABEL --metadata $OUT/application.json" \
  --kernel-name=RecurrentGatedDeltaRule \
  --launch-count=20 --warm-up=20 --aic-metrics=BasicInfo --output="$OUT"
python "$SCRIPT" --summarize "$OUT" --expected-count 20
```

核对每轮 application.json 的源码、case、安装对象和设备身份，保留完整目录。
比较各轮 mean、median、std、CV；下降应在独立轮次中复现并超过轮间波动。
性能持平或回退也按实记录，不能凭源码读取次数宣布收益。

如需验证瓶颈变化，按 [指标说明](msprof-metrics.md) 另采 PMU 与 PipeTimeline，
观察 Scalar 及计算前可见区间是否变化；诊断数据不与 BasicInfo 样本混合。
本机无 CANN/NPU，设备侧回传进度见下节；未回传的验收项目不记为通过。

## 5. 当前回传状态

已收到候选第 1 轮及 a6d33dbb 的补采基线，各有 20 条 BasicInfo。

| 指标 | 补采基线 | c031ed24 候选 |
|---|---:|---:|
| mean/us | 6.530560 | 6.583398 |
| median/us | 6.507823 | 6.659074 |
| CV | 4.6646% | 4.1759% |

候选均值高 0.81%、中位数高 2.32%，目前未显示收益。
两个相同基线二进制的轮次均值相差约 0.058 us，与本次均值差约 0.053 us 同量级，
当前按接近持平、典型用时略偏慢处理；一轮候选不足以认定稳定回退。
候选保留供追溯，不记录为已证实收益的优化；后续 head 遍历实验应从 a6d33dbb 独立比较。
设备精度、候选 PMU/流水结果尚未回传，详细统计与二进制核对见 [开发记录](validation.md)。
