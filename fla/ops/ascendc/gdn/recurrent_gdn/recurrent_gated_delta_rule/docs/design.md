# RecurrentGatedDeltaRule：A5 多头 decode

方案设计规则版本：V2。接口见 [api.md](api.md)，待验证事项见 [validation.md](validation.md)。
基线为 f918240b；参考 CHUNK_DEPENDENT/V1（a14961e9dfe8c8264768d67fb447101873da4ecf），
仅采用递推状态生命周期与事件核对方法。本算子沿用 AIV-only 实现，没有 Cube、CG 或跨核 workspace。

## 1. 目标

目标输入为用户确认的性能 case 1。保留其生成器、随机种子、值域、CPU 标杆、容差和全部 JSON。
实现两项可分别检视的改动：

1. A5 每个序列/每个 V 分片的末 token 调用不保存 updated FP32 scratch 的 VF 实例。
2. 为 case 1 所属的单序列、单 token、多头场景直接分配 head，省去 ComputeAvgload 和 head 归属扫描。

编码时没有设备基线或数值性能目标；首次用户反馈为 main 6.60 us、两项优化 6.71 us，
仅末 token 版本 93f5c70a 为 6.55 us，用户认为接近持平。新增调度当前未显示收益，
同步与运行时分派成本尚未分离，详见 validation.md。以同设备、相同安装与预热条件下的
msprof op / OpBasicInfo 的 Task Duration(us) 比较 f918240b、仅改动 1、改动 1+2。
Python 默认调用 20 次，各次独立 state 在首个 RGDR 前克隆；msprof warm-up=20、launch-count=20，
统计均值、中位数、标准差、P90/P99。输入构造与校验不计入目标 kernel。
case 1 理论每次少写 32*128*128*4=2 MiB UB，GM 读写量不减少；实际收益取决于 Store 是否在关键路径。

## 2. Stage 完整详设

令 L 为当前序列有效 token 数、hv 为 V head、G=Hv/Hk、hk=floor(hv/G)。
V 分片起点为 vi，有效行数 r=min(vStep,V-vi)。输入偏移为 (token*Hk+hk)*K 和
(token*Hv+hv)*V+vi；state 偏移为 index*stateStride0+hv*stateStride1+vi*stateStride2。
一般路径继续采用原按 L 加权的任务分配。新增调度只有一个序列、一个 V 分片，Nwork=Hv，
blockDim=min(Hv,可用 AIV 核数)，core c 处理 hv=c+j*blockDim，末尾 hv>=Hv 不执行。
case 1 的 Nwork=32；若可用核数>=32，则 blockDim=32、每核一个 head、一个 wave；
否则每核最多 ceil(32/blockDim) 个 head，wave=ceil(32/blockDim)。Hk=16 时 hk=floor(hv/2)。

| Stage | 单元 | 功能 | 主要空间 |
|---|---|---|---|
| 0 | Vector/MTE/Scalar | 元数据、gate/beta、QKV/state 搬入与转换 | 原输入队列、tmpBuff |
| 1 | Vector | gate state，计算 state-key dot，保存门控后 state | FP32 state scratch、attnTmp |
| 2 | Vector | delta、rank-one update、attention 投影及输出转换 | 寄存器、原输出队列 |
| 3 | MTE | 输出写回及队列释放 | 原 state/out GM 地址 |

### Stage 0：Vector，准备输入与分发

新增调度在原单 work 分支之后判断 api.md 中条件。先判断空核、prefix 和 length，再读取一次 state index。
head 整除关系已由 host 的 CheckShapeValueRangeAndRule 校验，AiCore 分派条件不重复执行 NV%NK。
该变体只去掉额外取余，保留后续 head 偏移中的必要除法和所有同步，不预设其性能收益。
每核调用原 CopyInGamaBeta(0,1)，沿用完整 head gate/beta 准备，不引入 Q/K 复用或缓冲调整。
每个本核 head 调用原 CopyInQKV、PrefetchState、LoadPrefetchedState。
在 UB 标量读取前发出一对 V_S Set/Wait，覆盖 gate Exp、beta/QKV Cast、query scale 和 state Cast。
空工作在事件和输出分配前退出。跨 head 按原 CopyOut 函数释放队列后再复用输入/输出空间。

### Stage 1：Vector，门控与第一遍归约

保持 FP32 Sg=exp(g)*S（以及通用路径可选逐 K gate）、dot=reduce(Sg*k) 的原指令顺序。
第一遍保存 Sg 和 dot，LocalMemBar<VEC_STORE,VEC_LOAD> 后第二遍读取，不能省去此 scratch。
g/beta、缩放和归约树与基线一致；不做 FMA 替换、代数重排或新的数值转换。

### Stage 2：Vector，更新与投影

保持 delta=beta*(v-dot)、Snew=Sg+delta*k、out=reduce(Snew*scaled_q)。
stateOut 转换和 attention 投影都直接使用更新后的 FP32 寄存器。
Compute<writeUpdatedScratch> 选择同一 VF 的编译期参数：
通用路径 seq_i+1<seq1 时 true，否则 false；新增调度固定 false，原单 work 默认 false。
true 实例逐操作保持原逻辑，false 仅删除第二遍 updated scratch Store。
末 token 后，下一 V 分片/下一 head 必先 LoadPrefetchedState 覆盖 scratch，所以不存在后续读取者。
两个原 VF LocalMemBar、所有输出 Store、队列入队和第一遍 scratch Store 保持。

### Stage 3：Vector/MTE，写回与释放

保留所有 token 的 state GM 写回，不能只保存序列最终状态。
新调度只有一个 token，按原 pending flush 的等价次序 CopyOutAttn -> CopyOutState。
state index、head stride、block stride 均取自原输入；不改变 tensor view offset 的调用层处理。
通用路径的 pending 输出与多 token 次序不变。

### 空间、搬运与同步汇总

L1/L0/算法 workspace=0，原 16 MiB 系统 workspace 不变。
UB 容量由原 platform/tiling 预算检查，所有 InitBuffer、restUbSize 与队列深度保持基线。
case 1 的 tmpBuff 相对字节区间为：

| 对象 | 半开区间 | 生命周期 |
|---|---|---|
| attnTmp | [0,512) | 两遍 dot/projection 与最终 out 转换 |
| v | [512,4608) | 输入 Cast 至当前 head VF 完成 |
| q | [4608,8704) | 输入 Cast/scale 至当前 head VF 完成 |
| k | [8704,12800) | 输入 Cast 至当前 head VF 完成 |
| FP32 state | [12800,78336) | 初始 Cast、两遍 VF；仅非末 token 保存更新供下一 token |
| beta | [78336,79360) | 一次 batch 搬入至本核 head 全部完成 |
| g | [79360,80384) | 一次 batch 搬入/Exp 至本核 head 全部完成 |

输入 Q/K/V 队列各 2048 B，BF16 state 输入 32768 B，g/beta 输入 1024/512 B；
state 输出每 slot 32768 B，attention 输出每 slot 256 B，slot 数沿原 host 选择的 1/2。
每个有效 head 仍读取/写回 BF16 state 各 32 KiB；第一遍 gated scratch 必须写回，
末 token updated scratch 少写 64 KiB/head。Q/K 在两个 V head 间重复搬运保持原状。

```text
检查有效任务 -> 原输入 EnQue/DeQue -> Cast/Exp/scale/state Cast
-> 新调度 V_S Set/Wait -> 读 g/beta 标量 -> VF 第一遍
-> 原 LocalMemBar -> VF 第二遍（末 token 不写 updated scratch）
-> 原 LocalMemBar -> 原输出 EnQue/DeQue -> GM 写回 -> Free -> 下一 head
```

R01/R20：AIV-only 继承原 head 所有权，新增直接映射覆盖每个 head 恰好一次，不引入跨核归约。
R03/R08/R11：地址、缓冲和峰值分配不变；新调度不持有跨 head 活跃输出。
R10/R16：只删除无消费者 Store，未合并或新增 GM 数据流。
R21：所有算术、Exp、Cast、mask 与顺序保持，末次 UB 结果不再读取；非有限值与精度仍需 NPU 验收。
