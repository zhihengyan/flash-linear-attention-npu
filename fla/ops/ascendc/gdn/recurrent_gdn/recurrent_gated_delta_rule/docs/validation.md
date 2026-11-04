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
