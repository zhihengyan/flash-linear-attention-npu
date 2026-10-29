# RecurrentGatedDeltaRule：A5 输入启动阶段融合实验

方案设计规则版本：`V2`。基线为 `fec1e18b`（scratch），实验分支为
`codex/gdn-a5-input-startup-fusion`。接口见 [api.md](api.md)，证据及验证见 [validation.md](validation.md)。

## 1. 目标

用户固定模型 case：A5、B=T=Nk=Nv=1、K=V=128、BF16 state、g-only、无 accepted-token。
原 perf case 0、输入分布、seed、dtype 和统计方法保持。目标 kernel 是实际命中的
RecurrentGatedDeltaRule BF16 single-work 实例；按 op_summary 的 Task Duration(us) 判断 kernel 收益，
其它 task 不合入。端到端另列；原预热和每轮 20 样本沿用，实际次数、版本、频率状态由内网日志归档。

scratch 最近四轮主 kernel 均值为 4.49/4.49/4.43/4.46 us，等权汇总 4.4675 us；端到端为
18.3365 us。state Cast 融合三轮均值 4.36/4.43/4.39 us，约 4.3933 us，端到端未见稳定改善。
本轮从 scratch 独立创建，不叠加该 Cast 融合、系数读取或 V_S 删除代码。
用户明确本轮是性能探索，设备精度不作为当前性能实验的前置；原精度规则仍保留，未经测试不宣称通过。

### 候选优先级和收益上限

| 候选 | 可作用的成本及代价 | 本轮选择 |
| --- | --- | --- |
| 小输入合包、一次准备 VF、state 提前搬入 | 合并 beta/g/Q/K/V 的 ready/free 组，并争取使 state DMA 与小张量 VF 重叠；覆盖整个输入启动路径，代价为显式打包地址及寄存器存活 | 第一优先级，实现一个整体输入启动策略 |
| 仅前移 PrefetchState | 只争取重叠，仍保留五个小输入队列和原准备调用 | 作用范围小于选定策略 |
| 全部输入合为一个 ready 组 | 少一个组，但小张量 VF 也要等完整 state DMA，失去独立重叠窗口 | 保留为参数化后备，暂不实现 |
| 继续 Cast 融合或单独 V_S 调整 | 已回传无稳定端到端收益，不能直接累加理论字节数 | 本轮不叠加 |
| 同一 head 增加核数 | 需改变当前按 HV 的映射及地址/分核设计，当前 Nwork=1 | 不在现有映射下直接采用 |

这里的“第一优先级”指当前支持范围内、已有结果尚未否定且覆盖最多输入启动成本的可落地候选，
不是已测得全局最大收益。一个主变量是 SINGLE_WORK 的输入启动路线替换；该路线包含配套的
打包、ready 边界和准备 VF，净收益不能单独归因到其中一个组成部分。

令 Msmall/Mstate 为小输入/state DMA 时间，Psmall 为融合后的小张量准备时间，H 为队列及发射成本。
新路线预计启动关键路径近似为 Msmall + max(Mstate,Psmall) + 原 state Cast + H_two_groups。
相对完全串行路线，重叠上限为 min(Mstate,Psmall)，另可能减少四组队列交接和准备 VF 启动。
基线已经存在的重叠、编译器自动融合和 SDK 内部同步会削弱这些收益，源码请求数不等于实际指令数。
没有可靠的分项时间下界，不能承诺收益超过波动；用户已授权以单独实验验证这一优先候选。
历史 main 端到端约 18.83 us，累计 5% 约需到 17.8885 us，即相对 scratch 再省约 0.448 us；
该历史换算不证明本轮可达成，也不替代同轮 main 对照。

## 2. Stage 完整详设

### 符号、工作域和 Stage 结果

沿用原 HV 分核及 host 选择谓词：A5、BF16、B=T=Nk=Nv=1、K=V=vStep=128、blockDim=1、
stateStride2=128、g-only、无 accepted-token。Nbase=1、HV=1、CG=1、Nwork=1、blockDim=1，
单 AIV 执行一个有效工作和一个 wave，无 head 尾块或跨核归约。其它模板、平台及调度保持 general。
P 为原 qInQueue slot 0 基址，I 为原 stateInQueue slot 0 基址，T 为原 tmpBuff 基址。
基址由未改的 InitLocalBuffers/TPipe 分配；以下半开区间均以字节为单位。所有队列相互独立。

| Stage | 执行单元 | 功能 | 主要空间 |
| --- | --- | --- | --- |
| 0 | Vector，一个显式准备 VF | beta 扩展、exp(g)、Q/K/V 扩展和 q scale；可与已发射 state MTE2 并行 | P、T 中原 q/k/v/beta/g 区域 |
| 1 | Vector，原 state Cast API | BF16 state 转为 FP32 scratch | I、T 中原 state 区域 |
| 2 | Vector，原两遍递推 VF | gate、dot、delta、更新、输出转换 | 原 FP32 scratch、attention tmp、输出队列 |

保留 Stage 0/1 分离以保留 state DMA 与小输入计算重叠窗口；Stage 1/2 分离沿用 scratch 基线，
不混入已试过的 Cast 融合。Stage 2 内两遍循环、两处 LocalMemBar、尾部 scratch 免写回和全部数学不变。

### 输入地址、搬运和 Stage 0

qInQueue 原容量为 MAX_MTP*128*sizeof(BF16)=2048 B，本轮复用它容纳一个 832 B 小输入 record：

| UB 半开区间 | 内容 / dtype / 大小 | 生产与消费 |
| --- | --- | --- |
| [P+0,P+256) | q，128 BF16，256 B | Q DMA -> 准备 VF load |
| [P+256,P+512) | k，128 BF16，256 B | K DMA -> 准备 VF load |
| [P+512,P+768) | v，128 BF16，256 B | V DMA -> 准备 VF load |
| [P+768,P+800) | beta，首 2 B 有效，其余 padding | beta DMA -> VF BRC_B16，仅读 2 B |
| [P+800,P+832) | g，首 4 B 有效，其余 padding | g DMA -> VF BRC_B32，仅读 4 B |
| [P+832,P+2048) | 未使用，不含有效数据 | 本 work 不读写 |
| [I,I+32768) | 原 BF16 state 128x128 | 原 PrefetchState DMA -> Stage 1 Cast |

所有小输入起点 32 B 对齐，q/k/v 各读 256 B，标量不读取 padding。五个 DMA 仍分别读取各自的
原 GM tensor，输入地址都为原偏移 0，不能将独立 GM tensor 当作连续空间。packed beta/g 的
typed LocalTensor 偏移分别按 BF16/FP32 元素数换算，不按字节误作元素数。

先分配 P 并发射 beta、g、q、k、v 五个 MTE2 Copy，再 EnQue(P) 发布小输入 ready；
然后调用原 PrefetchState(stateStride0*index,128)，发射 I 的 DMA/EnQue。
最后 DeQue(P) 并发射 Stage 0 准备 VF，返回后 Free(P)。I 的搬入必须早于 DeQue(P)，
且 I 不属于 P 的 ready 事件，否则无法形成预期重叠。MTE2 内的五次搬运仍串行，不能宣称五路 DMA 并发。

Stage 0 一次 VF 内完成 beta 的 BF16->FP32、g 的 FP32 Exp，以及 q/k/v 的 BF16->FP32。
Q/K/V 每次 load 128 个 BF16，以 Cast ZERO/ONE 扩展为偶/奇 K 的 64-lane FP32；k/v 交错 Store
到原 FP32 区域。q 保留寄存器，完成原 q*scale 两个 Muls 后交错 Store，避免额外 q UB 往返。
beta 和 g 使用首 lane mask、FIRST_ELEMENT_B32，仅写原输出区域的一个 FP32 值。
输入 Cast、Exp 和 q scale 的每个结果内部运算顺序不变；只重排独立数据的准备与保存。
不新增 FMA、Exp 算法选项或控制寄存器模式。有限 BF16 到 FP32 数学上精确扩展；q*scale 的
溢出范围和 g Exp 的下溢/溢出风险沿用原接口。原 CANN 9.1 符号已显示 g 使用 Reg Exp ZEROING，
本轮使用相同 Reg Exp 入口，但设备精度、subnormal 和最终生成指令没有在本机验证。

### Stage 1/2 的数据和资源

| UB 半开区间 | dtype / 逻辑大小 | 本轮生命周期 |
| --- | --- | --- |
| [T+0,T+512) | attention tmp，FP32，512 B | Stage 2 两遍 dot/output scratch |
| [T+512,T+4608) | v FP32 预留 4096 B，目标有效 512 B | Stage 0 Store -> Stage 2 load |
| [T+4608,T+8704) | q FP32 预留 4096 B，目标有效 512 B | Stage 0 scaled Store -> Stage 2 load |
| [T+8704,T+12800) | k FP32 预留 4096 B，目标有效 512 B | Stage 0 Store -> Stage 2 load |
| [T+12800,T+78336) | state FP32，65536 B | Stage 1 Cast -> Stage 2 第一遍读写 -> 第二遍读取 |
| [T+78336,T+78400) | beta FP32，64 B，首 4 B 有效 | Stage 0 Store -> 原 V_S/Scalar GetValue |
| [T+78400,T+78464) | exp(g) FP32，64 B，首 4 B 有效 | Stage 0 Store -> 原 V_S/Scalar GetValue |

未使用的 q/k/v/g/beta 原输入队列仍照原 InitBuffer 预留，不改变容量、地址、UB profile 或事件池配置。
stateOut/attnOut 的原独立队列、原 1/2 slot 深度和每槽 32768/256 B 保持，VF 写入后由原 MTE3 消费。
Stage 1 原 LoadPrefetchedState DeQue(I)、Cast、Free(I) 完全不改；Stage 2 主 VF、Compute、输出 helper 不改。
L1/L0/算法 GM workspace 为 0，原 16 MiB 系统 workspace 保留。新增 UB 为 0，P 的有效 record
832 B 小于原 2048 B；总预留峰值和最大连续空间需求均不增加。新 VF 寄存器峰值/spill 待 CANN 确认。

GM 逻辑读量仍为 state 32768 B、QKV 768 B、beta/g 6 B、原三个 INT32 metadata；输出仍为
state 32768 B、out 256 B。不减少 GM 数据量，本轮减少的是交接、准备启动和可能的关键路径等待。

### 同步和退出

| 资源 / slot | ready/free 闭环 | 单个有效 work |
| --- | --- | --- |
| P，原 qInQueue slot 0 | Alloc -> 五次 Copy -> EnQue -> DeQue -> 准备 VF -> Free | 各一次 |
| I，原 stateInQueue slot 0 | 原 Prefetch Alloc/Copy/EnQue -> 原 Load DeQue/Cast/Free | 各一次 |
| 原显式 V_S | state Cast 后 Fetch -> Set -> Wait -> Scalar g/beta | 一对，不删除 |
| 主 VF gated scratch、attention tmp | 原 Store -> 两处各自 LocalMemBar -> 原后续 Load | 完全保留 |
| 输出队列 | 原 Compute EnQue -> CopyOut DeQue/MTE3/Free | 完全保留 |

统一伪代码：检查 prefix/length -> 读一次 index -> Alloc(P) -> Copy(beta,g,q,k,v)
-> EnQue(P) -> 原 PrefetchState(I) -> DeQue(P) -> Stage 0 VF -> Free(P)
-> Stage 1 原 LoadPrefetchedState(I) -> 原 V_S Fetch/Set/Wait -> 原 Scalar g/beta
-> Stage 2 原 Compute -> 原 CopyOutAttn -> 原 CopyOutState -> 原 TPipe 退出。
P/I 均只有一个本 work，没有槽位回绕、下一次 Alloc 或跨核通知；空工作在全部搬入、队列和事件前退出。
不能 Free(P) 后再调用准备 VF，也不能用 V_S 替代任一 MTE2 ready。
队列组从 beta/g/q/k/v/state 六组变为 packed/state 两组是源码层面计数，不等于实际 event 指令减少四对。

同步依据见 [核内同步说明](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/intra_core_sync_overview.html)，
融合收益/代价见 [VF 特性说明](https://asc.gitcode.com/api/SIMD-API/c_api/reg_compute/key_features.html)，
g 的默认 Reg Exp 见 [Exp](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/basic_arithmetic/Exp-17.md)。
文档用于选择方向与 API，不能替代目标环境编译、实际流水重叠或设备结果。

R01/R20：映射和工作数不变；R03/R08/R11/R17：沿用独立原分配，P 子区间无重叠且生命周期闭环；
R10/R16：GM 请求数/字节不变，q scale 在寄存器复用；R12/R19：各计算阶段保留明确 VF 边界，
小输入和 state ready 分离用于重叠假设；R21：FP32 运算及原精度标准保留，本轮按用户要求不执行设备精度。
