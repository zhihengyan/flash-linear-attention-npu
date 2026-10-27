# A5 recurrent GDN 计时接入报告

当前状态（2026-10-27）：源码接入、静态检查与独立审核完成；CANN 编译及 A5 实测待内网完成。
分支 `codex/gdn-a5-stage-timer`，基线 main `c9e2f1db`。本轮只有 profiling，无优化改动。

## 接入范围

- 复用 op-timer-ascendc 的四个模板，定制为 AIV_ONLY linear core_id，4 个固定项、27 个动态项。
- 动态每类型最多 16 个样本、容量 128 核；metadata 保存真实次数和截断状态，按核独占 cache line。
- 普通 ACLNN/stable schema 保持；新增 Profile C 符号、私有 dispatcher 和内部 optional timer。
- 默认 perf case 0 的原 ATK 精度/性能入口以及原 build_inputs 均复用；普通/诊断同值输入对照。
- 原 VF、归约树、原屏障、tiling、UB、队列与 ViewCopy 保留，不混入实验 A/B。
- 各阶段 NoBarrier；仅诊断总窗口末尾一次 drain。A5 关闭自动 DCache 写回，显式发布本核记录。

## 本地检查

- `python -B tests/atk/recurrent_gated_delta_rule/scripts/test_timer_layout.py`：10 项 CPU 测试通过。
- 检查枚举/命名对应、27 对动态 Begin/End、raw 全零/错容量/缺固定点/缺动态配对/缺核/截断、
  默认 case 实际工作缺失、私有 adapter wiring、目标 `.o` 空集合/哈希不一致和 trace 时钟门禁。
- `python -B torch_custom/fla_npu/tools/stable_coverage.py --strict`：通过；全部公开入口保持完整，
  诊断入口单独精确列名并核对 wiring，不忽略其它未发布 adapter。
- `python -B torch_custom/fla_npu/tools/op_abi_parity.py`：39 adapters，0 mismatch；私有 schema 13 个参数。
- 使用 `op_abi_validate.py` 的解析/比较函数核对目标 Profile C 头和 stable forwarding：参数数量/类型匹配。
- `git diff --check`、Python AST/本地文档链接检查通过；移除打点后每条原 RGDR 语句和顺序相同，
  VF 与 InitLocalBuffers 逐字相同，原 tiling processor/struct、公共 ACLNN 头、公共 stable adapter 未修改。
- 主入口离线 CLI smoke 通过：合成 raw 生成 parsed CSV，状态为 offline_parse_only；无 torch/ATK/CANN 导入。
- 尚未执行 C++/CANN 编译；本地 Windows 未发现 CANN、torch_npu 或 C++ 编译器。
- Python 生成的测试数据只证明布局和解析规则，不是设备时间或 NPU 精度证据。

## 独立审核（skill Step 9）

独立审核者：fresh-context 子 agent `rgdr_timer_audit`，未参加实现、只读核对，最后增量复核通过。
本节反映其实际执行的源码/CPU 检查，不等同于设备通过。

| 审核项 | 独立证据 / 结果 |
| --- | --- |
| 数学步骤覆盖 | 完整 VF 与 HEAD 字符串比较 True；gate、dot、delta、rank-one、projection、cast 都在 RECURRENT_VF_ISSUE；明确是共用下发窗口，不伪造纯计算分项 |
| 原流程和数据生命周期 | 未改变队列、prefetch、pending 排空、tiling、UB、ViewCopy；普通/诊断共享 L0/同一 kernel；所有输入、state、timer 引用保留到 synchronize |
| 固定 Tik/Tok | KERNEL 入口行 57→71，INIT 58→66，PROCESS 67→69；AVGLOAD 在 arch35 行 330→334；分别使用独立起点状态，无覆盖 |
| 槽位 / 枚举 / 命名 / 清单 | 4 固定、27 动态、16 槽/类型、128 核；C++ / parser / flow 表四方逐项一致；动态索引见下表 |
| 模式及锚点 | 27 类均 Begin/End 同 type/slot、Overwrite；阶段起止 NoBarrier，DeQue 保留 ready 等待，只有总窗口末尾 Tok 排空，无新增 SyncAll |
| 核和发布所有权 | linear GetBlockIdx；每核事件行 880 int64、次数行 32、metadata 64；128B 分隔；核 0 独写头部；全部结束戳之后每核一次 Publish |
| 容量 | N_TIMING_COUNTER=112640，动态次数=4096，META_OFFSET=116736，HEADER_OFFSET=124928，TOTAL_BUFFER_SIZE=124944 int64；校验/解析一致 |
| 参数和原 ABI | Profile forwarding 为 q,k,v,beta,state,lengths,indices,g,gk,accepted,timer,scale,out；timer 为 def/L0/kernel 第 11 输入；原 ACLNN 头和原 stable adapter diff 为 0 |
| 原用例与精度 | 原 perf/accuracy spec 相同、复用原 build_inputs/mixed_tolerance_bm；普通/诊断额外 bitwise 比较，不新增容差 |
| 产物 / 实际路由 | 目标 `.o` 非空及全集哈希对拍、原 opapi/kernel OK、当前 source hash；缺真实工作计数或缺固定点一律拒绝 |
| 环境和状态副本 | ATK_ENV / sys.prefix 不同拒绝，无 atk 拒绝；CANN_ENV/FLA_NPU_ENV/FLA_NPU_OPP_ENV 不重复覆盖已准备 OPP；state 复制保持 shape/stride/storage_offset，warm_state 命名保活 |
| raw / trace | 头、容量、配对、核集合、实际调用/截断、负 duration、总窗口包含关系均检查；trace 需要明确频率和共同时钟 |

动态配对位置位于 `op_kernel/arch35/recurrent_gated_delta_rule.h`（Begin→End）：

| 索引 | 阶段 | 行号 |
| ---: | --- | --- |
| 0 | BATCH_GATES_ISSUE | 367→369 |
| 1 | HEAD_QKV_ISSUE | 589→591 |
| 2 | GK_COPY_WAIT | 403→405 |
| 3 | GK_EXP_ISSUE | 406→408 |
| 4 | Q_COPY_WAIT | 417→419 |
| 5 | K_COPY_WAIT | 420→422 |
| 6 | V_COPY_WAIT | 423→425 |
| 7 | QKV_CAST_ISSUE | 426→430 |
| 8 | QUERY_SCALE_ISSUE | 432→434 |
| 9 | STATE_PREFETCH_ISSUE | 442→449 |
| 10 | STATE_DEQUEUE_WAIT | 454→456 |
| 11 | STATE_CAST_COPY_ISSUE | 457→464 |
| 12 | TOKEN_PARAMS | 618→628 |
| 13 | OUTPUT_ALLOC_WAIT | 484→487 |
| 14 | RECURRENT_VF_ISSUE | 499→523 |
| 15 | OUTPUT_ENQUEUE | 524→527 |
| 16 | ATTN_DEQUEUE_WAIT | 532→534 |
| 17 | ATTN_COPY_ISSUE | 535→539 |
| 18 | STATE_OUT_DEQUEUE_WAIT | 544→546 |
| 19 | STATE_OUT_COPY_ISSUE | 547→552 |
| 20 | BETA_DEQUEUE_WAIT | 563→565 |
| 21 | BETA_CAST_ISSUE | 566→568 |
| 22 | G_DEQUEUE_WAIT | 575→577 |
| 23 | G_EXP_ISSUE | 578→580 |
| 24 | HEAD_TOTAL_ISSUE | 371→373 |
| 25 | SLICE_TOTAL_ISSUE | 603→656 |
| 26 | TOKEN_TOTAL_ISSUE | 617→648 |

实际执行：CPU unittest `-v` 10 项、strict coverage exit 0、schema parity 0 mismatch、
目标 Profile 头对拍、Python AST、diffcheck、上述 27/27 配对清单以及完整 VF 字符串对比。
初审发现两处 AIV_ONLY 模板替换后的孤立 else、固定点/实际工作缺失仍可成功、空 `.o` 集合
误通过三类问题，均已修复，并由审核者重跑通过。增量又修复旧 FLA_NPU_OPP_ENV fallback。

非阻断边界：直接 C Profile 调用者负责同设备/128B 对齐/不别名/清零，stable 才执行完整地址检查，
已写入 C 头与 api.md。全仓头校验工具的其它既有缺头及 SolveTri known-gap 不是本次新增；
目标 Profile 对拍通过。DCCI 实际可见性、编译资源、torch/NPU state 副本与计时开销仍待设备验证。

## 设备结果与后续

暂无本分支设备 raw/CSV/profiling，不生成阶段占比或瓶颈结论。
按 [timer_run.md](timer_run.md) 在内网以完整重建的同一套 wheel/OPP 采集。
首先阅读未携带 timer 的 `system_tasks.json` 和同输入比较结果，再解释诊断 issue/等待窗口。
若 ViewCopy 实际额外 task 明显，再评估输出路径；若主 kernel 占主导，结合 VF/等待数据再设计优化。
