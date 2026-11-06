# RecurrentGatedDeltaRule：A5 多头 decode

方案设计规则版本：V2。接口见 [api.md](api.md)，待验证事项见 [validation.md](validation.md)。
用例来源为 f918240b，当前内核对照为 a6d33dbb；参考 CHUNK_DEPENDENT/V1（a14961e9dfe8c8264768d67fb447101873da4ecf），
仅采用递推状态生命周期与事件核对方法。本算子沿用 AIV-only 实现，没有 Cube、CG 或跨核 workspace。

## 1. 目标

目标输入为用户确认的性能 case 1。保留其生成器、随机种子、值域、CPU 标杆、容差和全部 JSON。
当前实现以 a6d33dbb（kernel 同 93f5c70a）为基线，只缩小单序列、单 token、多头的 head 遍历范围。
元数据复用代码恢复为基线，历史提交和结果保留。本轮保留原 head 归属、VF、搬运和同步，
不使用此前独立的多头直接分发计算路径。

以同设备、相同安装与预热条件下的 msprof op / OpBasicInfo 的 Task Duration(us) 比较版本。
Python 默认调用 20 次，各次独立 state 在首个 RGDR 前克隆；msprof warm-up=20、launch-count=20，
统计均值、中位数、标准差、P90/P99。输入构造与校验不计入目标 kernel。
case 1 理论每次少写 32*128*128*4=2 MiB UB，GM 读写量不减少；实际收益取决于 Store 是否在关键路径。

本轮固定 Ascend950DT_9582、case 1、32 个任务块；补采基线均值 6.530560 us、
中位数 6.507823 us、CV 4.6646%。此前其他型号上的均值不参与本轮判断。
PMU 的 Scalar 平均占比为 54.49%，PipeTimeline 存在计算前和输出后的 Scalar 区间；
没有源码标签，不能将这些窗口全部归因于 head 扫描。
当前范围策略将 case 1 的源码级 head 归属检查从 32*32=1024 次减少到 32 次，
不改变浮点计算量、张量搬运或元数据读取次数，不新增除法或同步。
目标是在同环境重复对照中获得超过轮间波动的下降；两个基线轮次均值相差约 0.058 us，
这不是环境噪声上限。尚无单次归属检查的独占成本，不将整个采样窗口作为可回收时间；
实际收益以随后设备反馈为准。

用户随后回传：当前实现 0d188aca 相对 a6d33dbb936cf40528942133428e371889b5e21a，
三组交替 A/B 实验的性能收益约 1.9%，精度测试通过。据此采用本轮 case 1 的 head 范围策略。
2026-11-06，用户在另一张卡上复测，收益约 1.8%，确认保留此优化。
两次结果均为用户回传。复测设备与后续候选的证据边界见 [开发记录](validation.md)。

## 2. Stage 完整详设

令 L 为当前序列有效 token 数、hv 为 V head、G=Hv/Hk、hk=floor(hv/G)。
V 分片起点为 vi，有效行数 r=min(vStep,V-vi)。输入偏移为 (token*Hk+hk)*K 和
(token*Hv+hv)*V+vi；state 偏移为 index*stateStride0+hv*stateStride1+vi*stateStride2。
一般路径继续采用原 ComputeAvgload / IsCurrentBlock 按 L 加权的任务分配。
仅 B=1、L=1、Hv>1 时，以 a=avgload=ceil(Hv/blockDim) 计算本核连续区间
[headBegin,headEnd)=[c*a,min(c*a+a,Hv))；headBegin>=Hv 时不进入 head 循环。
它与原 IsCurrentBlock 每 a 个 head 推进一核的归属相同，不能改为 head=c+j*blockDim。
case 1 的 blockDim=min(Hv,可用 AIV 核数)，仍为原连续分配；不增加新 TilingKey 或独立计算路径。
case 1 的 Nwork=32；若可用核数>=32，则 blockDim=32、每核一个 head、一个 wave；
否则每核最多 ceil(32/blockDim) 个 head，wave=ceil(32/blockDim)。Hk=16 时 hk=floor(hv/2)。

| Stage | 单元 | 功能 | 主要空间 |
|---|---|---|---|
| 0 | Vector/MTE/Scalar | 元数据、gate/beta、QKV/state 搬入与转换 | 原输入队列、tmpBuff |
| 1 | Vector | gate state，计算 state-key dot，保存门控后 state | FP32 state scratch、attnTmp |
| 2 | Vector | delta、rank-one update、attention 投影及输出转换 | 寄存器、原输出队列 |
| 3 | MTE | 输出写回及队列释放 | 原 state/out GM 地址 |

### Stage 0：Vector，准备输入与分发

ComputeAvgload 和序列元数据读取与 a6d33dbb 完全一致，保留原前缀、长度和 accepted-token 校验。
在长度/prefix 校验后，默认遍历 [0,Hv)，仅 B=1、seqLen=1、Hv>1 时使用上述本核区间。
范围分支把 load=0、usedblk=c，等价于原扫描刚到 headBegin 前的私有游标；
区间内仍调用原 IsCurrentBlock 和同一循环体，不增加每 head 的模式判断。
该分支只有一个序列，区间结束后的私有游标没有后续消费者，不要求模拟尾部无工作扫描。
多序列、空序列、多 token 和单 head 都保留原扫描，B=0 仍不进入序列循环。
每个序列在本核首次有效 head 时读取 state index 并调用 CopyInGamaBeta。
ProcessHead、全部元数据读取和地址计算沿用 a6d33dbb，prefix、非零 state index、各级 stride、
gate、BF16/FP32 state 及 V 分片不作额外假设；accepted-token 非法值仍在搬入前退出。
每个本核 head 调用原 CopyInQKV、PrefetchState、LoadPrefetchedState，不引入 Q/K 复用或缓冲调整。
通用路径沿用原队列和同步；原 Hk=Hv=1 的 ProcessSingleWork 及其 V_S Set/Wait 保持。
当前 case 1 不进入 ProcessSingleWork，也没有新增多头调度的 V_S Set/Wait。

### Stage 1：Vector，门控与第一遍归约

保持 FP32 Sg=exp(g)*S（以及通用路径可选逐 K gate）、dot=reduce(Sg*k) 的原指令顺序。
第一遍保存 Sg 和 dot，LocalMemBar<VEC_STORE,VEC_LOAD> 后第二遍读取，不能省去此 scratch。
g/beta、缩放和归约树与基线一致；不做 FMA 替换、代数重排或新的数值转换。

### Stage 2：Vector，更新与投影

保持 delta=beta*(v-dot)、Snew=Sg+delta*k、out=reduce(Snew*scaled_q)。
stateOut 转换和 attention 投影都直接使用更新后的 FP32 寄存器。
Compute<writeUpdatedScratch> 选择同一 VF 的编译期参数：
通用路径 seq_i+1<seq1 时 true，否则 false；原单 work 默认 false。
true 实例逐操作保持原逻辑，false 仅删除第二遍 updated scratch Store。
末 token 后，下一 V 分片/下一 head 必先 LoadPrefetchedState 覆盖 scratch，所以不存在后续读取者。
两个原 VF LocalMemBar、所有输出 Store、队列入队和第一遍 scratch Store 保持。

### Stage 3：Vector/MTE，写回与释放

保留所有 token 的 state GM 写回，不能只保存序列最终状态。
通用路径按原输出队列配置立即写回或 pending flush；单 token 时仍先 attention、后 state。
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
-> 原通用路径同步 -> 读 g/beta 标量 -> VF 第一遍
-> 原 LocalMemBar -> VF 第二遍（末 token 不写 updated scratch）
-> 原 LocalMemBar -> 原输出 EnQue/DeQue -> GM 写回 -> Free -> 下一 head
```

R01/R20：AIV-only 继承原 head 所有权与次序，只跳过不属于本核的单 token head 扫描，不引入跨核归约。
R03/R08/R11：地址、缓冲和峰值分配不变，不持有跨 head 活跃输出。
R10/R16：只删除无消费者 Store，未合并或新增 GM 数据流。
R21：所有算术、Exp、Cast、mask 与顺序保持，末次 UB 结果不再读取；用户已回传精度通过，
专项覆盖与正式验收状态按 validation.md 中的证据范围记录。

当前采用 head 遍历范围，元数据复用已从实现中移除。
新增范围计算的成本已纳入本轮用户回传的 A/B 比较；当前实现保持，后续改动独立验证。
复现步骤和覆盖要求见 [实验说明](head-range-experiment.md) 与 [开发记录](validation.md)。
