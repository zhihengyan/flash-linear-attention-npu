# RecurrentGatedDeltaRule：A5 VF 逐行融合

方案设计规则版本：`V2`。接口见 [`api.md`](api.md)。本文记录既有 AIV_ONLY 实现及
本轮受影响的 Vector Stage；本轮不重新设计原有分核、TilingKey、队列和 UB 预算。

## 1. 目标

在 main 的 `571d7acf` 基线上实施用户指定分析中的实验 A，目标硬件为 A5（`ascend950`）。
原 VF 第一遍读取 state、施加 gate、归约并写回门控 state 和 dot，第二遍重读后更新 state
和输出。本轮让一行 state 的两个 FP32 寄存器跨上述操作存活，消除中间 state/dot 的 UB 往返。
实际序列长度为 1 时，另外省去无消费者的 FP32 递推 scratch 写入。

调用链继续为 Python stable wrapper → Stable ABI → ACLNN → AIV_ONLY kernel →
`arch35/RGDR::ProcessHead` → `ComputeRecurrentGatedDeltaRuleVF`。A2/A3 使用另一个 header。
不引入实验 B 的 tiling、缓冲深度调整、行交错、FMA 或公式重排。

用户已明确本轮只要求编码和静态检查；A5 在内网。本轮不设模型性能目标，也不把理论流量
减少作为实测提速。基线和改后版本的 A5 编译、精度、内存及性能验收留待设备环境执行。

## 2. Stage 0：Vector，单 token 的递推与投影

### 任务、公式和顺序

令 `L=seq1-seq0`、`K=128`、`R=vStep`、`r=curSingleV<=R<=128`、`s=sizeof(stateType)`。
一次逻辑任务是一个有效 sequence/value head；沿 V 轴按既有 R 分片。同一片内 token 按序执行，
每个 token 一次 VF。head 的 Q/K 映射为 `hk=hv/(Nv/Nk)`。
调度仍是当前 `IsCurrentBlock` 的按 token 数累积分配，`blockDim=min(B*Nv, availableAIV)`。
未改成 AIC/AIV 协同或 grid-stride；不将此既有调度描述为已满足新的分核方案。

例如功能核对 `B=1,L=1,Nk=8,Nv=32`（非用户模型 case），在 availableAIV≥32 时有
32 份任务、blockDim=32，每核 1 个 head，无任务尾部；R=64 时每个 head 两个 V 分片。

逐行保持以下 FP32 运算顺序，其中 q 已在 `CopyInQKV` 中乘以 scale：

```text
Sbar = S
if g:  Sbar = Sbar * exp(g)
if gk: Sbar = Sbar * exp(gk)
dot = ReduceSum((Sbar0 * k0) + (Sbar1 * k1))
delta = (v - broadcast(dot.lane0)) * beta
Snew0 = Sbar0 + delta * k0
Snew1 = Sbar1 + delta * k1
out = ReduceSum((Snew0 * q0) + (Snew1 * q1))
stateOut = cast_state(Snew)
attnTmp[row] = out
```

FP32 state 的 0/1 为连续两个 64 元素半行；BF16 state 沿用原有 FP32 scratch 的
`DIST_DINTLV_B32` 偶/奇拆分和 `DIST_INTLV_B32` 存储，保持两类路径各自的归约树。
门控乘法、Mul/Add 分离、FP32 累加、输出 BF16 RINT 转换均保持原顺序。
`ReduceSum` 只保证首 lane 有效，用 `Duplicate(dotBroadcast,dotSum,mask)` 在寄存器中广播。
仓内 `chunk_kda_bwd_finalize` 的 A5 VF 已使用相同的 MicroAPI 写法；广播语义参见
[CANN Duplicate 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/data_padding/Duplicate-48.md)。

`ProcessHead` 用已经从设备 metadata 得到的 `L==1` 选择 `Compute<false>`，其它有效 L
选择 `Compute<true>`。模板参数 `saveState` 透传到全部 gate 特化：false 时编译掉 FP32
scratch 写入，true 时每 token 保存 Snew 供后继 token 使用。两者均逐 token 写 stateOut，
输出与 state 槽位可观察行为不变；不增加 D2H 或 `.item()`。

### 地址、资源和生命周期

L1/L0 和算法 GM workspace 为 0，系统 workspace 仍按现有 host 请求，未用于本轮中间量。
所有 GM、TPipe 分配和 tmpBuff 地址保持基线；寄存器存活时间延长，不新增 UB 对象。

以下以字节为单位描述当前 InitLocalBuffers 的名义连续地址，`a(x)=ceil(x/32)*32`。
`dS/dO` 是现有 host 选出的输出缓冲深度，`G/GK` 是对应 gate 是否存在的 0/1。
TPipe 自身管理预留以实际 CANN 分配为准，本轮不据此扩大容量或提高 R。

| 绝对半开区间 | 对象、份数、布局 | 首次生产 / 最后消费及复用 |
| --- | --- | --- |
| `[0,2048)` | q queue，1 份，BF16 `[8,128]` | MTE2 / Cast，FreeTensor 后复用 |
| `[2048,4096)` | k queue，同上 | 同上 |
| `[4096,6144)` | v queue，1 份，BF16 `[8,128]` | 同上 |
| `[6144,U1)`，`U1=6144+128*R*s` | state input queue，1 份 `[R,128]` | MTE2 / FP32 scratch 装载，FreeTensor 后预取下一片 |
| `[U1,U2)`，`U2=U1+G*32*Nv` | g queue，可选 1 份，FP32 `[8,Nv]` | MTE2 / Exp，FreeTensor 后复用 |
| `[U2,U3)`，`U3=U2+GK*4096` | gk queue，可选 1 份，FP32 `[8,128]` | MTE2+Exp / 当前 head 所有片 VF，head 结束释放 |
| `[U3,U4)`，`U4=U3+a(16*Nv)` | beta queue，1 份，BF16 `[8,Nv]` | MTE2 / Cast，FreeTensor 后复用 |
| `[U4,U5)`，`U5=U4+dS*128*R*s` | state output queue，dS 份 `[R,128]` | VF / MTE3，FreeTensor 后复用 |
| `[U5,P)`，`P=U5+dO*a(2*R)` | attn output queue，dO 份 BF16 `[R]` | VF / MTE3，FreeTensor 后复用 |
| `[P,P+4*R)` | attnInUb，1 份 FP32 `[R]` | VF 逐行 out 标量 / 同 VF 最后打包，下一 token 复用 |
| `[P+4*R,P+4*R+4096)` | vInUb，1 份 FP32 `[8,128]` | Cast / 当前 head 所有片 VF |
| `[P+4*R+4096,P+4*R+8192)` | qInUb，同上 | Cast+scale / 当前 head 所有片 VF |
| `[P+4*R+8192,P+4*R+12288)` | kInUb，同上 | Cast / 当前 head 所有片 VF |
| `[P+4*R+12288,E)`，`E=P+4*R+12288+512*R` | stateInUb，1 份 FP32 `[R,128]` | 输入装载；L>1 时每 token VF 更新 / 当前片最后一次 VF |
| `[E,E+b)`，`b=4*ceil(8*Nv/16)*16` | betaInUb，1 份 FP32 | Cast / 本核当前 sequence 的 head VF |
| `[E+b,E+2*b)` | gamaInUb，1 份 FP32，g 缺失时仍预留 | Exp / 同上 |
| `[E+2*b,P+restUbSize)` | tmpBuff 剩余区，无活跃数据 | 仅分配器预留，不供本轮复用 |

每个 queue slot 的实际地址为该区间起点加 `slot*a(single_buffer_bytes)`；深度不变。
单 token scratch 在 stateOut 生成后失效，下一个 V 片先装入新 state 再读 scratch。
多 token scratch 从输入装载存活到当前片最后一次 VF，不与 attnTmp 等区间别名。
最高命名地址 E+2*b、总空闲和连续尾部空闲均不变；需在设备验证中核对实际 UB 分配，
本轮不声称已重新验证原 host 对 A5 的预算。

### 搬运与同步

每 head 的逻辑 GM state 读写仍为各一次 `128*128*s` 字节。Q/K、V、beta、g/gk、metadata
的 GM 次数、L2 策略、state 预取和所有 queue 操作均保持基线。

| 每个 state 元素的 VF 逻辑 UB 流量 | BF16 state | FP32 state |
| --- | ---: | ---: |
| 原两遍循环 | 18 B | 20 B |
| 融合，saveState=true | 10 B | 12 B |
| 融合，saveState=false | 6 B | 8 B |

上表不含初始 Cast/Copy、QKV、标量 scratch、寄存器溢出或 GM 搬运，不代表整核提速比例。
dot 的 UB store/load 和两遍循环之间的 VEC_STORE→VEC_LOAD 屏障一起删除。
逐行 out 仍写 attnTmp，最后打包前的 `LocalMemBar<VEC_STORE,VEC_LOAD>` 必须保留。

| 同步资源 | 作用范围 / 生产与消费 | 本轮变化 |
| --- | --- | --- |
| 输入 queue EnQue/DeQue/FreeTensor | 单 AIV MTE2→V，逐 slot | 无；state 装载完成后可预取下一 V 片 |
| gk 初始化 V_MTE2 event | 单 AIV，Duplicate→DataCopyPad | 无；继续使用现有 Set/Wait 闭环 |
| 输出 queue EnQue/DeQue/FreeTensor | 单 AIV V→MTE3，dS/dO slot | 无；保留 pending 输出排空 |
| VF 内 LocalMemBar | 单 AIV VEC_STORE→VEC_LOAD，attnTmp | 由两个减为一个，保护最终输出打包 |
| 跨核 flag / Mutex | 无跨核共享中间量 | 不新增 |

```text
空序列：沿用 Process 跳过，无 queue 获取
每 head：装入 QKV/gates；预取首片 state
每 V 片：DeQue state → Cast/Copy scratch → FreeTensor；如有下一片，预取它
  每 token：Alloc state/attn 输出
    L==1: VF<saveState=false>；L>1: VF<saveState=true>
    VF: 每行完整递推 → 写 out 标量和 stateOut
        → LocalMemBar → 打包 attnOut
    EnQue 输出；沿用即时 CopyOut 或消费上一 pending slot
  排空当前片 pending 输出，再进入下一片
head 结束：释放 gk queue tensor；所有已发布输出都已消费
```

### 数值、边界与评审

CPU 标杆的 gate→点积→delta→rank-one→投影顺序与上述实现一一对应。删除的是 FP32
等值 store/load，不改变 exp、计算 dtype、归约拆分、cast 时机或无效区 mask。
不增加乘加融合，不改变既有溢出/下溢和 NaN/Inf 的输入约束。
实际 L 判断涵盖 prefix、零长度和混合长度；accepted 槽位选择、GVA、稀疏 state index、
非连续视图及输出 mask 均沿用既有代码。保留全部 gate 模板，包括原有不可公开调用的无 gate 实例。

本轮静态设计评审：R02/R10–R19/R21 检查的是上述逐行数据依赖、地址不别名、同步及数值顺序；
R03/R08 不扩大既有分配；R01/R04–R07/R09/R20 中涉及 AIC/Cube 及任务重映射的方案
不在本次实验 A 内引入。寄存器压力、编译器调度和实际收益需要 A5 测量，不能由本设计判定。
回退时恢复此 A5 header 的基线 VF 和 Compute 调用即可，ABI 与 tiling 数据没有迁移步骤。
