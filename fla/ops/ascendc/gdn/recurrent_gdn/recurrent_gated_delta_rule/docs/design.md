# RecurrentGatedDeltaRule：A5 UB 与输出缓冲实验 B

方案设计规则版本：`V2`。公开契约见 [`api.md`](api.md)。本轮从 main `571d7acf` 出发，
保留其完整 A5 VF、数学顺序和任务调度，只修改 A5 host UB/缓冲 profile 的选择。

## 1. 目标

共享 host 预算按 A2/A3 的两个 FP32 矩阵和两个 FP32 行向量核算，A5 实际只有一个
FP32 state scratch 和一个 attention 行向量；另外 A5 始终分配 FP32 gamaInUb。
本轮按 A5 `InitLocalBuffers` 的全部存活对象计算，而不是从共享系数直接删掉一个矩阵。
UB 容量使用平台查询值，不硬编码 A5 为 256 KiB。

调用链为 stable Python → Stable ABI → ACLNN → host tiling processor →
arch35 AIV_ONLY kernel。host 用平台 `ASCEND950` 标记选择 A5 预算；缺省标记为 false。
快速拉起 example 当前显式包含通用 A2/A3 kernel，继续用 false，不把 A5 预算配给通用 kernel。
公开 schema、ACLNN ABI、传入设备的 tiling struct、TilingKey 和 workspace 大小不变。

合法候选仍为 `(stateOut,attnOut)=(1,1)/(1,2)/(2,2)`。先最小化 V 分片数，
同分片数在 T>1 时沿用较深缓冲优先。T=1 时只选 (1,1)：不会有第二个有效 token
与第一个 token 的输出重叠，既有 ProcessHead 在每个 V 片结束前排空 pending 输出。
T>1 即使实际是纯 decode，也不据 T=B 推断单 token。MAX_MTP=8 的预留仍保留。

本轮沿用用户已指定的编码和静态检查范围。实验 A 的用户反馈为 18.7685→19.3954 μs，
回退 0.6269 μs（3.34%）；没有本地原始日志、重复测量或指令 profiling，不能据此证实
寄存器 spill、UB/HBM 瓶颈或测量噪声。B 的目标是独立验证资源策略，不承诺提速幅度。

## 2. Stage 0：Vector，既有递推和投影

### 任务、数据流与候选

每个 sequence/value head 是原有逻辑任务，Q/K 的 `hk=hv/(Nv/Nk)`；分核仍使用原
IsCurrentBlock 的 token 负载分配，blockDim=min(B*Nv,availableAIV)。本轮不改为
AIC/AIV 协同、CG 分组或 grid-stride。以仓内 perf case 0 为功能例，B=T=Nk=Nv=1，
K=V=128，g-only/BF16 state，只产生一份 head 任务、启动一个 AIV；它不代表模型吞吐。

Stage 内输入 Cast/Exp → 按 V 片装载 FP32 state → 逐 token 两遍 VF → state/attn 输出
完全沿用 main，FP32/BF16 的归约树、门控顺序和 cast 不变。只改变 R=vStep、dS/dO
及 tmpBuff 的分配字节数，未引入实验 A 的行内融合。GM state 仍每 head 读一次，
每 token 写一次 `128*128*sizeof(state)`；分片改变不会减少 state 总 GM 字节数。

### 精确 UB 布局

单位为 Byte；a(x)=ceil(x/32)*32，s=state 元素字节数，R 为 8 的倍数且不超过 128。
G/GK 表示 gate 存在。q/k/v 保留 8 token，物理 K/V stride=128。

| 绝对半开区间 | 对象 / 份数 | 生命周期与复用 |
| --- | --- | --- |
| `[0,2048)` | q queue / 1，BF16 | MTE2→Cast→FreeTensor |
| `[2048,4096)` | k queue / 1，BF16 | 同上 |
| `[4096,6144)` | v queue / 1，BF16 | 同上 |
| `[6144,U1)`，U1=6144+128*R*s | state input queue / 1 | MTE2→scratch Cast/Copy→FreeTensor，再预取下一片 |
| `[U1,U2)`，U2=U1+G*32*Nv | g queue / 可选 1 | MTE2→Exp→FreeTensor |
| `[U2,U3)`，U3=U2+GK*4096 | gk queue / 可选 1 | MTE2→Exp→当前 head 全部片 VF→释放 |
| `[U3,U4)`，U4=U3+a(16*Nv) | beta queue / 1 | MTE2→Cast→FreeTensor |
| `[U4,U5)`，U5=U4+dS*128*R*s | state output queue / dS | VF→MTE3→FreeTensor，每 slot 独立 |
| `[U5,P)`，P=U5+dO*a(2*R) | attn output queue / dO | 同上 |
| `[P,P+4*R)` | attnInUb / 1，FP32 | 每 token VF 的 dot/out 标量→最终输出打包 |
| `[P+4*R,P+4*R+4096)` | vInUb / 1，FP32 | Cast→当前 head 全部片 VF |
| `[P+4*R+4096,P+4*R+8192)` | qInUb / 1，FP32 | Cast/scale→当前 head 全部片 VF |
| `[P+4*R+8192,P+4*R+12288)` | kInUb / 1，FP32 | Cast→当前 head 全部片 VF |
| `[P+4*R+12288,E)`，E=P+4*R+12288+512*R | stateInUb / 1，FP32 | 每片输入装载→逐 token 递推→片末失效 |
| `[E,E+b)`，b=4*ceil(8*Nv/16)*16 | betaInUb / 1，FP32 | Cast→当前 sequence 的本核 head VF |
| `[E+b,E+2*b)` | gamaInUb / 1，FP32，始终预留 | Exp→同上；g 缺失时不读但仍分配 |
| `[E+2*b,UB)` | 无活跃 tensor | 至少 128 B 管理/对齐余量，其余保持未分配 |

每个 queue slot 按 a(single_buffer_bytes) 分配，tmpBuff 字节数为
`4*R + 12288 + 512*R + 2*b`，尾端恰好覆盖 gamaInUb。
候选可用条件为 `queueBytes + tmpBytes + 128 <= queriedUbBytes`。
128 B 余量沿用原预算的保留量；实际 TPipe 管理行为仍需目标 CANN 验证。
各驻留区不重叠，tmpBuff 内 offset 与 main 相同。L1/L0、算法 GM workspace 为 0，
系统 workspace 仍请求原来的 16 MiB。本轮不扩大高于平台查询值的地址空间。

每个 profile 从 R=128 以 8 为步长递减，找到最小可行分片数后，沿用 main 的
`R=align8(ceil(128/分片数))` 平衡切片，避免大首片加小尾片。不同 profile 先比较
ceil(128/R)，然后比较 dS+dO，再比较 R；T=1 排除深缓冲。
例如假设 UB=256 KiB、Nv=32、g-only、FP32 state：R=128 时 (1,1) 的命名对象为
219392 B，加余量为 219520 B；(2,2) 为 285184+128 B，不能容纳。
新策略能选 (1,1)/R=128，原共享策略在此例 R=64。这是容量推导，非设备结果。
对 BF16 最小 perf 例，main 可能已经 R=128，B 主要改变输出深度与分配开销，收益可能很小。
若 R 变大为单片，原片间预取也失去后继片，必须测量净收益。

### 同步、边界与数值评审

| 资源 | 作用核、生产/等待/释放 | 本轮影响 |
| --- | --- | --- |
| 输入 queue slot | 单 AIV，MTE2 EnQue→V DeQue→FreeTensor | 顺序/深度不变 |
| state/attn output slot | 单 AIV，V EnQue→MTE3 DeQue→FreeTensor | T=1 深度为 1，不存在跨 token pending；T>1 保留原深度候选 |
| gk V_MTE2 event | 单 AIV，初始化后 Set/Wait→MTE2 | 不变 |
| VF 内两个 LocalMemBar | 单 AIV，gate/dot 中间读写及最终输出打包 | 不变，VF 与 main 完全一致 |
| 跨核 flag / Mutex | 无共享中间量 | 不新增 |

```text
host: 查询 SoC/UB → A5 枚举完整预算并选择 profile，否则运行原共享策略
kernel: InitBuffer 使用 profile；每 head 先装 QKV/gates 并预取首片 state
每 V 片: 装载 scratch，释放输入 slot；如有下一片则预取
  每 token: 分配输出 → 原两遍 VF → EnQue
    深度 1: 当前输出立即 DeQue/CopyOut/Free
    深度 2: 消费上一 pending slot，当前 slot 留待下一 token 或片末
  片末排空全部 pending，再进入下一片；head 末释放 gk
```

T=1 的充分条件不读取 metadata，不增加 D2H；空序列、prefix、accepted 初始 state、
GVA、非连续 state、全部有效 token 快照和重复调用沿用原处理。T=B 但有 `[0,2]`
等混合长度时仍按多 token 候选选择。UB 不足以容纳最小 R=8 时返回 host 错误。

R21：VF 计算、FP32 scratch 和输出转换均未改，资源变化不增加数值风险；不同 R 的
计算行相互独立，K 归约维与运算顺序不变。R03/R08/R11/R12/R17 按上述完整地址账本
检查，R01/R20 中的任务重映射不在本轮引入。采用收益和实际分支/slot 复用证据留待 A5 验证。
回退只需恢复 host A5 标记与预算选择，不涉及数据或 ABI 迁移。
