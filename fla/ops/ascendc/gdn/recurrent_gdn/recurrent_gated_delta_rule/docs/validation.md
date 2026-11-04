# A5 多头 decode 开发记录

基线 f918240b，原分支 codex/gdn-k32-tail-scratch-elision 保持该提交。
新分支 codex/gdn-a5-multihead-tail-scratch 从其创建，性能 JSON 原样继承。
本机为 Windows，无 torch/ATK/CANN/NPU 或 C++ 编译器；本轮交付编码和离线检查，设备验收待执行。
采用 CHUNK_DEPENDENT/V1，来源 a14961e9dfe8c8264768d67fb447101873da4ecf；
仅参考递推 scratch 生命周期、任务与事件闭环，不移植其 Cube/GM workspace 架构。

| 改动 | 静态验证 | 设备验证 |
|---|---|---|
| case 1 JSON | 对照原分支 Git blob | 运行 case 1；原文件保持不变 |
| msprof op 脚本 | case 1 加载、参数校验、CSV 过滤/计数/统计、模拟调用边界 | A5 真实采集，核对目标任务/样本数 |
| 末 token scratch | true/false VF 展开、四个 gate 分支、普通/末 token 分派 | BF16/FP32、g/gk/both、多 token、多 V 分片 |
| 多头单 token 调度 | head 归属/偏移、guard、空工作、V_S 顺序与 Set/Wait 配对 | case 1、非零 index、padded block/head、核数少于 head |
| 原路径 | 原单 work、UB、Copy helpers、公共 ABI/A2/A3 与基线比较 | case 0、原 200 精度场景及 MSS/确定性 |

性能 case 1：20 次独立 state 调用，msprof warm-up=20、launch-count=20、BasicInfo；
只统计 RGDR Task Duration(us)，与相同条件下基线交错比较。原来未提供该 shape 的设备基线或目标幅度。
理论末次 Store 少 2 MiB UB/调用，不据此换算延迟或吞吐改善。

开发顺序：先提交脚本/基线准备，再提交末 token 改动，最后提交多头调度。
保留中间提交以便在设备上分别归因两项优化。静态检查通过不代表设备精度/性能通过。

已完成：测评脚本 12 项离线检查通过，包括当前 runtime 返回多个 OPP vendor 的适配、
首次目标调用前全部 state 克隆及 CSV 计数/单位/过滤负向检查。
末 token 改动的 4 项源码检查通过：VF 数学保持、四个 gate 分支参数、仅末 token 分派变化、
UB/搬运与公开接口/A2A3 源码保持。尚未执行 CANN 编译或设备精度/性能测试。

最终编码后的检查：

- `python -B -m unittest discover -s tests/atk/recurrent_gated_delta_rule/scripts -p 'test_*.py' -v`：
  共 20 项，19 通过，1 项生产控制函数 C++ harness 因缺少编译器跳过。
- 源码检查能拒绝错误保留尾部 Store、遗漏 V_S Wait、head 归属重叠、错误 Q/K 分组四种变体。
- 原分支仍为 f918240b；三份 ATK JSON 的 Git blob 和 prefill K32 实现均与该基线一致。
- `check_issue_coverage.py`：188 个 completed issue、21 条规则映射通过。
- 四层静态检视脚本仍返回 42 条 DFX-MESSAGE error，0 warning；逐条确认所在 host/API 文件与
  f918240b 完全一致。这不是“检视零告警”。代表位置为 aclnn 实现 118 行、tiling 45 行及
  tiling_processor 86 行的旧 guard/通用 OP_LOGE；本轮不修改其公开参数报错。
- #362 的 stride/state 路径按原地址公式核对，#640 的设备精度/非有限值回归待执行，
  #641 的公开异常校验仍由未修改的 host/API 负责。
- `git diff --check` 与新增文档本地链接检查通过。以上均不代表硬件同步、机器码或性能已验收。

独立测量版本：7acfe2e1（脚本与设计准备、kernel 同 f918240b）、93f5c70a（仅末 token 省写）、
本分支最新提交（再加多头直接分发）。case 1 需从实际 host 日志确认 vStep=128 和 blockDim，
同一 TilingKey 不能证明命中了新运行时分支。

## 2026-11-04：首次性能反馈

用户通过本分支 msprof op 脚本测量并报告：main 平均 6.60 us，当前两项优化版本平均 6.71 us。
差值为 +0.11 us，约 +1.67%；当前观察没有显示收益。用户说明收益见效前尚未进行精度验证。
设备侧实际 commit、安装产物哈希、count、median/std/CV、频率及重复轮次未提供，
用户随后确认两次使用同一 msprof_op_rgdr.py、相同命令行参数且均为 --case-id 1。
具体 JSON/产物哈希尚未提供；不能将均值差直接归因为某一项改动或认定为随机波动。

源码复核：VF 数学、InitLocalBuffers、CopyInGamaBeta、CopyInQKV、LoadPrefetchedState 保持基线。
额外调度路径增加了运行时条件判断与 V_S 完成等待，该等待位于 state Cast 之后；
其同步代价及对 Scalar/Vector 重叠的影响是待测假设，不能在缺少精度与时序证据时直接删除。
2 MiB UB Store 是 32 个 head 的累计量；若每核一个 head，则每核省 64 KiB，
多个核并行执行，累计流量不等价于主 kernel 时延按相同比例下降。

下一轮优先测中间提交 93f5c70a（仅末 token 省写），与 main、9a515a70（两项优化）交错比较。
保持 JSON、脚本、设备、calls/warm-up/launch-count、安装校验流程一致，保存每轮 summary 和 application。
若 93f5c70a 改善而 9a515a70 退化，先定位调度分支；若 93f5c70a 也退化或无收益，
再检查 Store 是否在关键路径及编译分派代价。本轮记录反馈，不继续叠加未经验证的 kernel 改动。

## 2026-11-04：消融结果与重复取余实验

用户切换到 93f5c70a 并重新编译，报告平均 6.55 us，认为与 main 基本持平。

| 版本 | 用户回传均值 | 相对 main |
|---|---:|---:|
| main | 6.60 us | 基线 |
| 93f5c70a，仅末 token 省写 | 6.55 us | -0.05 us，约 -0.76% |
| 9a515a70，再加多头调度 | 6.71 us | +0.11 us，约 +1.67% |

多头调度版本比仅省写版本多 0.16 us，约 +2.44%。这组消融提示优先检查新增调度开销，
仍缺少重复轮次/波动与流水指标，不能据此认定 V_S 是唯一原因。精度尚无设备验证结论。

源码发现新增 CanUseSingleTokenHeads 在 AiCore 重复计算 NV%NK==0，host 早已拦截 Nk=0 和
Nv%Nk!=0，且共享 tiling processor 在生成 kernel 参数前执行这些检查。
下一变体仅从设备分派谓词删除这项重复取余，保留 Nk>0 防护、完整多头执行函数、V_S Set/Wait、
VF、搬运和输出逻辑。它是单变量实验，不是已经验证有效的性能修复。
非整除组合仍由原 host 拒绝，不改变公开支持范围或错误语义。

对应源码检查新增 host 检查保持、设备无重复取余、多头执行函数与 9a515a70 一致的断言；
C++ 可选 harness 只对 host 合法的 head 组合验证设备优化谓词，不把不合法 tiling 当作有效入口。
本机 9 项 kernel 检查中 8 项通过、1 项因缺少编译器跳过；新变体的 CANN 编译和设备性能仍待执行。

## 内网执行

先为相应提交编译并安装单算子 run 包，并使用新 Python 进程：

```bash
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
./build_out/fla_npu_linux-x86_64.run --install
python scripts/verify_libcust_opapi_md5.py --run-package build_out/fla_npu_linux-x86_64.run
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=mssanitizer
```

run 包路径按实际架构替换。原 MSS 文件尚不能代表全部新运行时分支，补齐代表输入与设备选择证据后
才能记为正式验收。case 1 的精度须单独验证，同一输入调用 CPU 标杆与未被 profiler 重放的 NPU 各一次。
全部有效结果通过后再做 README 的 msprof op 采集；硬件故障、超时或精度失败按仓库精度定位路由处理。
NPU 验证完成前保留本记录，不归档“阶段 5 已完成”。
