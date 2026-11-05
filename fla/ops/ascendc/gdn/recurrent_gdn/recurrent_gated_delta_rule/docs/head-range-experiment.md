# A5 单 token head 范围实验

本轮内核以 a6d33dbb 为基线，相对它只修改 ProcessGeneral 的 head 遍历区间。
元数据复用不带入本轮，历史 c031ed24 及结果保留供追溯。prefill K32 和末 token scratch 省写沿基线。

## 1. 为什么选择这项

同环境 PMU 中 Scalar 平均占比为 54.49%，流水也显示计算前及输出后的控制区间。
此前元数据复用未观察到收益，因此本轮改查通用 head 扫描。
这些流水区间没有源码标签，不能认定全是扫描时间，也不能把它们全算作预期收益。

case 1 为 B=T=1、Hk=16、Hv=32、BF16 state、g-only，Block Dim=32。
原来每核扫描 32 个 head，实际只处理其中 1 个；新实现每核只扫描它负责的区间。
源码级归属检查由总计 1024 次减为 32 次，VF、张量/元数据访问、队列与同步均保持。

本轮沿用同一 ProcessHead，不复制独立计算路径，不新增 V_S 事件、除法或 TilingKey。
与此前单独的多头直接分发实验不同，本轮保留原连续归属、原逐 head 调用与同步位置。

## 2. 范围公式与边界

只有原长度/prefix 校验通过，且 B=1、seqLen=1、Hv>1 时使用范围：

```text
a = avgload = ceil(Hv / blockDim)  # 已由原 ComputeAvgload 算好
headBegin = core * a
headEnd   = min(headBegin + a, Hv)
```

原 IsCurrentBlock 在单 token 时每累计 a 个 head 才推进一核，因而核 core 恰好负责这个连续区间。
进入区间前置 load=0、usedblk=core，等价于原扫描刚到 headBegin 前的私有游标；
区间内仍执行原 IsCurrentBlock。headBegin>=headEnd 时不进入循环，元数据/队列也不额外访问。
该范围分支只有一个序列，结束后的私有游标没有后续消费者，无需模拟无工作的尾部扫描。

| Hv / 核数 | 原归属及新范围 |
|---|---|
| 32 / 32 | 各核一个 head |
| 32 / 5 | 前四核各 7 个，末核 4 个 |
| 5 / 4 | 2、2、1、0 个；末核无工作 |

不能替换成 core+j*blockDim 的轮转分配。多序列、多 token、空序列及单 head 走原扫描。
prefix、非零/稀疏 index、accepted、各级 stride、gate、BF16/FP32 state、多 V 分片均保留原处理。
所有输出释放和 kernel 收尾保持；不能把原尾部 Scalar 循环当成可删除同步的依据。

## 3. 离线验证

```bash
python -B -m unittest discover -s tests/atk/recurrent_gated_delta_rule/scripts -p 'test_*.py' -v
```

范围测试从生产 C++ 提取整数表达式，对照独立的旧游标扫描，枚举 Hv=2..256、核数=1..96，
共 1,187,280 个核范围，验证每核归属和次序一致，包括不能整除及无工作核。
源码检查确认仅 ProcessGeneral 与 a6d33dbb 不同，metadata、VF、缓冲和同步不含上一轮改动。
若 g++/clang++ 可用，还会编译抽取的旧/新 Scalar 控制函数，比较调用地址/顺序、gate 参数和
metadata 访问，覆盖 prefix、非法长度、accepted、多 token/多序列回退及不同输出深度。
该 C++ 检查在缺少编译器时明确跳过，不能代替 NPU 精度、流水或实际分支验证。

## 4. 重新编译安装

以下命令在上一轮 Ascend950DT_9582 环境、仓库根目录执行。新候选必须重新安装，不复用旧 .o：

```bash
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
```

按构建输出选择当前架构 run 包。以下以 x86_64 为例，其他架构替换路径：

```bash
./build_out/fla_npu_linux-x86_64.run --install
python scripts/verify_libcust_opapi_md5.py --run-package build_out/fla_npu_linux-x86_64.run
```

若使用已有的 wheel 构建流程，同样需重新构建并安装本轮产物，验证实际对象再启动新 Python 进程。
基线使用 a6d33dbb 对应产物，不使用 c031ed24；分别保留两套包与安装校验结果。

## 5. 设备验证和采样

先独立验证 case 1 精度，profiler 重放输出不作精度结论：

```bash
(
  cd tests/atk/recurrent_gated_delta_rule
  atk node --name npu_dut --backend npu --devices 0 \
    node --name cpu_golden --backend cpu \
    task -c atk_recurrent_gated_delta_rule_perf.json --task accuracy --bm_device cpu \
    -p executor_recurrent_gated_delta_rule.py -s 1 -e 2
)
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=accuracy
```

原 200 条精度用于回归；同时按 [开发记录](validation.md) 和 [ATK README](../../../../../../../tests/atk/recurrent_gated_delta_rule/README.md)
补齐 MSS、确定性、所有可达分支和实际选择证据。TilingKey 本轮没有变化，不能仅凭 key 名证明命中范围分支；
需要结合已安装对象、实际 B/Hv/blockDim、设备序列值及执行证据核对。

在仓库根目录采集 BasicInfo，基线安装后 LABEL=baseline，候选安装后 LABEL=head_range：

```bash
LABEL=head_range
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

保持同机、同软件、相同参数，各轮输出新目录，交错运行基线与候选；各轮分别统计 mean/median/std/CV。
最近基线均值为 6.530560 us、中位数 6.507823 us。已有两轮基线均值相差约 0.058 us，
这不是噪声上限，不能只与一个历史均值相减就认定小幅收益。
需要时按 [指标说明](msprof-metrics.md) 另采 PMU/PipeTimeline，看 Scalar 控制区间是否相应变化，
诊断时延不与 BasicInfo 混合。

当前仅有编码和离线检查；本机无 CANN/NPU，实际设备分支、精度和性能尚未验证，不预设一定加速。
