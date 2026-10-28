# A5 单 work 实验开发记录

基线 main `c9e2f1db`，目标分支 `codex/gdn-a5-single-work-fastpath`。
用户授权仅编码和本地静态检查；A5/CANN 在内网，本地无 CANN、torch 或 C++ 编译器，不请求重新提供 SSH。
本文件保留到内网验收完成，不能把源码检查记为 NPU 精度或性能通过。

## 当前实验

设计见 [design.md](design.md)，接口边界见 [api.md](api.md)。只改变单 work 标量控制和元数据复用。
不引入 A/B 或 profiling 分支代码，不更改 public ABI、CPU 标杆、200 条 ATK 精度用例或精度标准。
MSS / determinism 保留原 case 0，选取已有精度 case 1/2 补齐 FP32/general 与 BF16/general。

| 项目 | 证据 / 状态 |
| --- | --- |
| main 参考 | 用户 main/B 交错实验 main 均值 4.89、4.85、4.87 μs，合并 4.87 μs |
| 目标 | 相同原 case、环境、预热/采样方式下主 kernel 均值/中位数稳定下降，内网待测 |
| 模板选择 | 声明 BF16/general、FP32/general、BF16/single 三个组合，入口传参与 host key 参数一致；源码检查通过，设备实际选择待验证 |
| VF/UB/同步 | 原 VF、InitLocalBuffers、Copy/Compute/ProcessHead/IsCurrentBlock 函数体及 general 调度与 main 相同，源码对照通过 |
| 公共接口与原输入 | 原 ACLNN 声明/实现、OpDef、tiling struct 字段、CPU 标杆、200 条精度 JSON、perf JSON 和生成器对照 main 通过 |
| 本地脚本 | `test_single_work_fastpath.py -v`：6 项，5 通过，1 因无 C++ 编译器跳过 |
| 测试资产与文档 | Python AST、精度 200 / 性能 1 / MSS 3 条 JSON 与 case_spec 一致性、新增 Markdown 链接检查通过 |
| C++ 选择/控制语义测试 | 脚本会编译生产选择头和提取的单 work 控制函数；本机未运行，不算已通过 |
| 格式检查 | `git diff --check` 通过 |
| 实际 CANN 编译 | 未执行 |
| 原 ATK 200 条精度 | 未执行 |
| 非连续/空工作/accepted/gk/FP32/A2A3 回归 | 未执行设备验证 |
| 默认 case 性能与 PMU | 未执行，不宣称收益 |

采用仓库 CHUNK_DEPENDENT/V1 参考（来源 a14961e9dfe8c8264768d67fb447101873da4ecf），
仅参考递推状态生命周期、tiling 模板与同步核对方法；本例没有 Cube 或跨核 workspace。

## 实现追溯与静态检查边界

| 设计项 | 实现位置 | 本地证据 / 剩余验证 |
| --- | --- | --- |
| A5 与 shape/gate/stride 条件 | `op_host/recurrent_gated_delta_rule_single_work.h`、tiling `DoLibApiTiling` | 三组合、入口参数对照通过；选择谓词的 C++ 执行和真实 tiling 日志待测 |
| 单 work、元数据复用 | arch35 `ProcessSingleWork` | 不含任务循环和新同步；prefix/length 先检查，state index 只读一次；行为测试待编译运行 |
| 回退调度 | arch35 `ProcessGeneral` | 函数体与 main `Process` 一致 |
| Stage 数学、搬运、队列和释放 | 原 VF、InitLocalBuffers、Copy/Compute | 函数体一致；生成机器码、自动同步及 NPU 精度待测 |
| 公开 ABI / tiling 数据 | 原 OpDef、aclnn、packed struct | 源码与 main 一致；新私有模板参数需重建 host 与 kernel，不能只替换旧 `.o` |
| 三个模板的 ATK 代表 | 精度 0/1/2、MSS 0/1/2；性能 0 | JSON 映射通过；实际 key/blockDim/命中证据待采集 |

无设备时只能确认编写的控制路径和原 helper 的源码边界。编译器是否裁掉常量分支、
自动同步是否改变、标量指令是否减少，必须从 A5 编译产物和 PMU 证明。
新专用分支的空工作、非零 state index、padded block stride 也须在设备上回归；
现有精度 200 条只在 case 0 覆盖新分支，不能把其它 shape 的回退验证当作这些边界的设备证据。

## 内网最小验证命令

已有当前环境 wheel 时，只需目标算子 run 包；本轮没有新 stable adapter 或接口，不必重编全部 wheel：

```bash
python -B tests/atk/recurrent_gated_delta_rule/scripts/test_single_work_fastpath.py -v
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
./build_out/fla_npu_linux-x86_64.run --install
python scripts/verify_libcust_opapi_md5.py --run-package build_out/fla_npu_linux-x86_64.run
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=mssanitizer
PERFORMANCE_START=0 PERFORMANCE_END=1 \
  bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -npu_device_id=0 -scope=performance
```

run 包路径按实际架构选择；安装后使用新 Python 进程。确认产物比对实际包含目标 `.o`，不接受跳过比对。
范围是 `[start,end)`，默认 case 使用 0/1。精度必须有实际用例和通过证据，ATK 退出 0 不代替结果检查。
性能只取 OP Type=RecurrentGatedDeltaRule 的 Task Duration，不混入 Neg/Cast/有限性检查。
main/C 交错复测；需要模板命中证据和 Scalar/Vector/MTE 分项，不将 timer 分支 5.9341 μs 当作 main 基线。
每轮沿用此前 20 个 RGDR 样本；实际预热次数、ATK/CANN 版本和频率状态从内网配置/日志记录，
本机没有这些配置，不为它们填入推测值。
