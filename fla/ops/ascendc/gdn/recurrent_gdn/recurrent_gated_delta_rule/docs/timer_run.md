# A5 recurrent GDN profiling 操作说明

使用本分支 `codex/gdn-a5-stage-timer`，main 基线 `c9e2f1db`。
唯一主要入口是 `tests/atk/recurrent_gated_delta_rule/scripts/profile_performance.py`。
沿用原 perf JSON 默认 case 0 和原 executor，不新增替代模型输入或精度容差。
流程及窗口含义见 [timer_flow.md](timer_flow.md)，当前状态见 [timer_report.md](timer_report.md)。

## 1. 进入内网已有容器和环境

本地会话没有内网容器名、Linux 挂载路径或 CANN/conda 位置，以下变量必须填已有环境的真实值，
不会替用户猜镜像、创建容器或安装 CANN。若已经在容器中，从第二段开始。

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
: "${RGDR_CONTAINER:?set the existing A5 container name}"
docker exec -it "$RGDR_CONTAINER" /bin/bash
```

在容器内，先激活你已有的独立 conda 环境（延用前两轮实验的环境），然后：

```bash
: "${RGDR_REPO:?set the absolute repository path inside this container}"
cd "$RGDR_REPO"
: "${CANN_ENV:?set the absolute existing CANN set_env.sh path}"
source "$CANN_ENV"
command -v python
python -m pip --version
npu-smi info
git branch --show-current
git rev-parse HEAD
```

核对实际芯片为 A5；CANN、Python/torch/torch_npu 版本保持与基线同一套。不要混装 A2 wheel。
若设置 ATK_ENV，必须先激活它，使其与当前 python 的 sys.prefix 相同；当前解释器需已有原 ATK 包。
采集入口对子进程沿用父进程已激活的 CANN/已准备的 OPP，不重复 source 可选的 CANN_ENV/FLA_NPU_ENV。
Windows 路径不能直接当作容器内路径。采集脚本会记录实际包、设备名、加载产物及环境。

## 2. 首次完整重建并安装

本轮新增了私有 stable adapter 与内部 optional timer，首次需要重建 wheel，不能只覆盖 kernel。
按仓库开发者指南使用算子筛选，重建目标算子的 host/proto/kernel 和 wheel 中的 stable adapter。
无需清空 `FLA_NPU_OPS` 后编译全仓算子。构建不修改 VF 数学或 tiling 策略。

```bash
FLA_RGDR_ENABLE_TIMER=1 FLA_NPU_SOC=ascend950 FLA_NPU_OPS=recurrent_gated_delta_rule python scripts/build_wheel.py
```

### 已有构建失败时先保留现场

`gmake: *** [Makefile:156: all] Error 2` 只说明依赖目标失败，没有给出目标名或编译器诊断。
并行构建中其它目标仍可能随后输出 `Built target`。先保留本次完整日志
`dist/fla-npu-build.log`（指定 `--wheel-dir` 时在相应目录），不要再次启动 wheel 构建覆盖现场。

若完整日志仍只有顶层失败状态，在同一个已激活 CANN 的 shell、仓库根目录下，使用现有 CMake
构建目录串行续编并合并 stdout/stderr。该命令不运行 `build.sh` 的清理步骤，已有且仍有效的目标可复用。
这是定位失败目标的命令，不会完成 wheel 的后续打包或安装；若失败重现，将生成明确的目标/命令日志。

```bash
test -f build/CMakeCache.txt || { echo 'missing existing CMake build tree'; exit 1; }
set -o pipefail
FLA_RGDR_ENABLE_TIMER=1 LD_LIBRARY_PATH="$PWD/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  cmake --build build --target package --parallel 1 --verbose \
  2>&1 | tee rgdr-build-resume.log
```

定位并修复失败后，再执行上面的单算子 wheel 命令。不能把局部 kernel `Built target` 或静态检查
当作完整 wheel 构建通过。

2026-10-28 已定位并在源码修正 `Invalid DFX:aclnnRecurrentGatedDeltaRuleGetWorkspaceSize`：
原 L2 DFX 宏误放在公共 helper，现已回到对应 API。若内网保留了失败现场，可以先仅复编日志中
失败的 host 对象目标验证这条错误，不必先重编全部 kernel：

```bash
FLA_RGDR_ENABLE_TIMER=1 LD_LIBRARY_PATH="$PWD/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  cmake --build build --target ophost_transformer_opapi_obj --parallel 1 --verbose
```

此目标通过只代表 host 对象编译完成，之后仍需完成 package/wheel 构建与安装。

使用构建末尾打印的**准确 wheel 和 run 包**，设置它们的绝对路径；不要按 commit 文件名排序选旧包：

```bash
: "${RGDR_WHEEL:?set this build's exact wheel path from the build output}"
: "${RGDR_RUN_PACKAGE:?set this build's exact fla_npu_linux .run path}"
test -f "$RGDR_WHEEL" && test -f "$RGDR_RUN_PACKAGE"
python -m pip install --force-reinstall --no-cache-dir --no-deps "$RGDR_WHEEL"
```

后续只迭代 kernel/host 打点且已安装本轮完整 wheel 时，按仓库标准覆盖内嵌 OPP：

```bash
FLA_RGDR_ENABLE_TIMER=1 bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
# 将 RGDR_RUN_PACKAGE 更新为此轮构建实际生成的准确路径。
"$RGDR_RUN_PACKAGE" --install
```

每次安装后使用新 Python 进程。主要入口会自动调用
`scripts/verify_libcust_opapi_md5.py --run-package "$RGDR_RUN_PACKAGE"` 并要求 `.o` 比对证据，
还会把已加载 stable `.so` 的 source hash 与当前 checkout 比较；exit 0/WARN 不算内核一致性通过。
构建后保留本轮 build 中的 kernel `.o`，供一致性校验。
若 kernel 根目录不同于默认 build/lib/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/kernel，
主入口显式传 `--built-kernel "$RGDR_BUILT_KERNEL"`。目标算子的对象集合必须非空且哈希一致；
不会将“0 个对象相同”当作验证成功。

## 3. 一个入口执行原测试和同输入采集

```bash
python tests/atk/recurrent_gated_delta_rule/scripts/profile_performance.py \
  --device 0 --case-id 0 --run-package "$RGDR_RUN_PACKAGE"
```

入口先在导入 torch/torch_npu 前准备已安装 wheel 的 OPP 环境，再执行：

1. 原 ATK 精度入口，只运行与 perf spec 完全一致的 accuracy case，标准仍为 mixed_tolerance_bm。
2. 原 ATK performance_device 入口，统计方式、预热/采样由原 ATK 保持，不在此另造计时循环。
3. 从原 executor 的 build_inputs 生成一份真实输入，保存 inputs.pt；普通/诊断各用同值 state 副本。
4. 私有 Profile 入口写独立清零 timer；严格比较普通/诊断 out 和完整 state（bitwise 相同）。
5. 同步后先保存 raw 与 outputs，再验证布局/核数/配对/截断、生成 CSV、metadata 和汇总。
6. 汇总原系统 profile 的 op_summary Task Duration，帮助区分主 kernel、ViewCopy 或其它 task。

Profile 仅单 launch；原 perf 若改成 repeat_calls>1，本入口明确拒绝，需先设计多 launch 采集。
普通/诊断调用各有 60 秒 watchdog（保守地包含 host 提交与 synchronize，不能称为已确认 kernel
入场的计时）；超时子进程退出，主进程保存失败状态与日志。检查 npu-smi 确认资源释放后再重试。
这不是全 200 条用例正式验收，不应将一次默认 case 采数标记为全量精度通过。

默认只输出 cycles，不假设时钟。核实目标计数器频率且确认相关 AIV 共同时钟后，可在同一主入口增加：

```bash
python tests/atk/recurrent_gated_delta_rule/scripts/profile_performance.py \
  --device 0 --case-id 0 --run-package "$RGDR_RUN_PACKAGE" \
  --cycles-per-us "$RGDR_VERIFIED_CYCLES_PER_US" --shared-clock
```

此时额外生成 `rank_0_npu_time.json`，可在 https://ui.perfetto.dev/ 或 MindStudio Insight 打开。
核间时钟未核实时保留 CSV；不要从不同 launch 的计数器 origin 推断调用间隔。

## 4. 阅读结果与离线解析

默认输出在 `profiling_results/recurrent_gated_delta_rule/<UTC_run_id>/`，已加入 gitignore。

- `status.json` 必须为 captured_and_compared；failed 时先看日志，raw 不因解析失败而删除。
- `artifact_check.log`、`atk_accuracy.log`、`atk_performance.log`、`capture.log` 保存每步证据。
- `system_tasks.json` 为未携带 timer 的原性能调用 task 统计，判断 ViewCopy 是否实际形成独立任务。
- `capture.json` / `timer_metadata.json` 记录实际 vStep/dS/dO、每核每类型真实调用次数及截断。
- `rank_0_npu_time.csv` 为完整 raw；`rank_0_npu_time_parsed.csv` 为每核每阶段 start/end/duration。
- `inputs.pt` / `outputs.pt` 保存同输入对照；计时器不会把这些导出放进原性能循环。

阶段名称中的 ISSUE 是含标量下发的窗口，不能当作独立纯计算耗时。DeQue 窗口是原 ready 等待；
父子窗口有嵌套，禁止简单相加。KERNEL_TIMING 含诊断末尾 drain，但不含 DCCI 发布开销，
不能用它代替未打点主 kernel 的 msprof Task Duration。

至多使用一个离线入口，仍复用同一脚本，不依赖 NPU/torch/ATK：

```bash
python tests/atk/recurrent_gated_delta_rule/scripts/profile_performance.py \
  --parse-raw "$RGDR_RAW_CSV" --output "$RGDR_NEW_PARSE_DIRECTORY"
```

输出目录必须不存在，以免覆盖已有采集。离线成功状态为 offline_parse_only，不声称执行设备验证。

## 5. 打点开销对照和回归

原 ATK performance 使用不携带 timer 的常规入口。进一步检查打点编译是否影响普通调用时，
可分别用 FLA_RGDR_ENABLE_TIMER=0/1 构建并按相同方式安装 run 包、核对产物、新进程执行原性能入口：

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -soc=ascend950 -scope=performance
```

ENABLE_TIMER=0 的产物不能用于主采集入口：诊断无有效 raw，解析必须失败。
不为分阶段计时逐运算加 barrier，也不把带 timer 的任务耗时直接拿来宣称优化收益。
后续回归需覆盖 BF16/FP32、g/gk/both、accepted/varlen、同核多 work、slot 采样截断与所有早退发布。
