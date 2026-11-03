# ChunkGatedDeltaRuleFwdPrepare 设计

方案设计规则版本：V2

## 1. 目标

接口与支持范围以 [api.md](api.md) 为准。本算子在 Ascend 950 上以 MIX AIC:AIV=1:2 融合
GDN 前向准备链路，chunk 之间无状态依赖。task 按 value-head 组成 pack，AIV0/AIV1 分别
处理 task 0/2 与 1/3，ping/pong 槽由 task id 决定；GVA owner 只计算一次 Q/K norm 与 KKT。
`G=HV/HK=3/4` 保持每 pack 3/4 个 task；`G=1/2` 可按负载选择更短的 pack，但不得拆开同一
HK 的 value-head 共享组。

本轮推进 Prepare 内部 Stage7 和小 case pack 调度，不改变公开接口、数学语义、输出舍入顺序
或支持范围。Stage3 masked Mul／Store 候选已实测为负收益并回退，实验数据仅保留在验证记录中，
不与 Stage7、短 pack 混合归因。

固定性能环境：Ascend 950、device 2、相同 ATK 性能 JSON 与统计入口。性能结果为端到端
`chunk_gated_delta_rule_fwd` 两条 case 的正式中位数；每项至少复测两轮，收益须超过基线波动。

| Case | Shape | 正式基线 | 三轮范围 | 本轮目标 |
| --- | --- | ---: | ---: | --- |
| 推理 case 0 | `B=1,HK=16,HV=32,T=11274,K=V=128,BT=64`，varlen + final state | 1705.298 µs | 1703.162～1710.482 µs | 精度通过且稳定下降 |
| 训练 case 1 | `B=2,HK=HV=32,T=8192,K=V=128,BT=64`，定长无状态输出 | 2707.735 µs | 2704.416～2709.092 µs | 精度通过且稳定下降 |

基线波动分别为 7.321/4.676 µs。历史均值 1704.700/2704.500 µs 仅用于交叉验证。
Stage7 候选减少 A 的重复 LoadA 并合并 W/U MMAD，直接作用于现有 Prepare 计算路径，
但最终收益必须设备实测，不预写百分比。

## 2. Stage 0–7 完整详设

### 任务域与资源

task 是 `(sequence-chunk, hv)`，工作总量为 `chunkCount*HV`，`hk=hv/(HV/HK)`。
固定 `BT=64`。`G=1/2` 的候选 pack 从 `G` 到原 pack 大小按 `G` 递增，选择使
`ceil(ceil(totalTasks/pack)/numCore)*pack` 最小者，平局保留较大 pack。该式表示最忙核的
task 数；它只在确实存在核间负载差时缩短 pack，不减少总计算量，也不把“增加 pack 数”
本身视为收益。所有 AIC/AIV 循环均按实际 `cur.nThis` 执行，初始 Stage4 credit 也按实际
pack 大小发布。

Stage4 Y 使用每核 64 KiB GM workspace，四个 16 KiB slot，除此之外不新增 workspace。
L1 可用 512 KiB，当前最高使用地址 496 KiB；每 AIV UB 可用 248 KiB，V=128 当前最高使用
地址 236 KiB。L0A/L0B 各 64 KiB，L0C 256 KiB。本轮不增加任何片上常驻量。

| Stage | 执行单元 | 功能与主要结果 |
| --- | --- | --- |
| Stage0 | AIV | 生成 VCS identity、scatter index、L1 零叶块及 Cube I64 |
| Stage1 | AIV | gate/beta、可选 Q/K L2Norm；k' 写 L1，其余结果写 GM |
| Stage2 | AIC | `KKT=k' @ k'^T`，L0C Fixpipe 到 AIV UB |
| Stage3 | AIV | gate 下三角、`-L=KKT⊙G`、叶块求逆、`-L` 与叶块写 L1 |
| Stage4 | AIC | `Y=I+LeafLeft@(-L)`，经 GM workspace 回读 L1 |
| Stage5 | AIC | `A=LeafLeft+Y@LeafRight`，按需输出并保留 L1 A |
| Stage6 | AIV | 生成 `vb=v*β`、`kbg=k'*β*exp*(g')`，写 L1 |
| Stage7 | AIC | `W=A@kbg`、`U=A@vb`，写 UB 后由 AIV drain 到 GM |

### Stage0：Vector， resident auxiliary

UB `[0.5,1) KiB` 为 scatter index，`[1,9) KiB` 为 FP32 VCS identity；临时 FP32 I/zero 使用
`[25,41) KiB`。L1 四个 LeafLeft/Right slot 先清零，AIV0 生成 `L1[480,496) KiB` 的
FP32 NZ I64。所有 copy 由 V/MTE3 事件闭合后才释放临时 UB。

### Stage1：Vector，输入准备

每 AIV 的 g'/beta 位于 UB `[9,10) KiB` 四个 256 B bank；Q/K 输入使用 `[10,74) KiB`，
QHat/KHat 使用 `[75,140) KiB`，rstd 使用其间的 256 B bank。owner 的 k' 以 BF16 NZ C0=16
写入 `L1[64,128) KiB` 对应 task slot。tail 先补零，再覆盖有效 M 行；g/beta 仅写有效 M。
k' 发布 Stage1 ready 后才允许 Stage2；跨 pack 覆写前等待前一 pack Stage4 释放 NegL/k' 别名。

### Stage2：Cube，KKT

每个 HK owner 执行一次 `KKT[64,64]FP32=k'[64,128]BF16 @ k'^T[128,64]BF16`。
A/B 来自同一 L1 k' slot；偶/奇 task 使用 L0A/L0B `[0,16)/[32,48)` 与
`[16,32)/[48,64) KiB`，L0C 使用 `[0,64)/[64,128) KiB`。M_MTE1 保护输入，M_FIX 后
Fixpipe 为 UB `[140,156)/[156,172) KiB` 的 FP32 ND KKT，再发布 FIX_M 与跨核 ready。

### Stage3：Vector，下三角与叶块求逆

每 AIV 两个 task bank 的 UB 布局为：

```text
ping: LPacked [10,18) KiB；ResVcs [18,34) KiB；LFull [34,50) KiB
pong: LPacked [50,58) KiB；ResVcs [58,74) KiB；LFull [74,90) KiB
KKT:  [140,172) KiB 两个 16 KiB FP32 ND bank
```

`GateLowerLVF` 计算 `G[i,j]=-beta[i]*exp*(clip(g[i]-g[j],-50,50))`（`i>j`），
并在同一全行 Store 中把对角线和上三角显式写零。等待 KKT 后，`MulKktGateVF` 读取完整
两行，对全部 64 列执行 FP32 Mul 与全行 Store，得到 `-L=KKT⊙G`。PackDiagLeaves、
UB→L1 NZ 与 VCS 输入布局均保持原实现。

`-L` 写入 `L1[64,128) KiB` 对应 task 的 FP32 NZ slot；两个 32×32 逆叶块分别写入
LeafLeft/Right 的 L1 slot。V_MTE3 与 MTE3_V 保持原闭环，Stage3 完成后由 MTE3 发布
跨核 ready。tail 的 padding 行仍为零，不改变同步参与方。

### Stage4：Cube，块递推第一段

`Y[64,64]FP32=I@I+LeafLeft@(-L)`，两个 MMAD 在同一 L0C 累加。I 预填可与 Stage3
重叠；Y 经 `gmWsY[core][task]` 的 16 KiB FP32 NZ slot Fixpipe 后 MTE2 回读至
`L1[0,64) KiB`。每个 task 使用独立 L0C 64 KiB stride 与 workspace slot，FIX/MTE2 完成后
才释放并通知下一 pack 可复用 k'/NegL。

### Stage5：Cube，块递推第二段

`A[64,64]FP32=I@LeafLeft+Y@LeafRight`。L0A/L0B 前两组 bank 与 S7 bank 分别承载两次
MMAD，输入为 L1 I、LeafLeft、Y、LeafRight。完整且 `outputA=true` 时经 GM ND 输出并回读
L1 BF16 NZ A；否则直接 Fixpipe 到 `L1[256,288) KiB` 对应 task 的 BF16 NZ A。tail 只写
有效 M 行到 GM，L1 保留完整 padded tile。

### Stage6：Vector，W/U 右操作数

`vb=v*β`，`kbg=k'*β*exp*(g')`，均为 BF16 ND→L1 NZ C0=16。K/V 临时 UB 分别位于
`[32,64)` 与 `[64,128) KiB` 的 ping/pong bank；g'/beta 继续驻留 `[9,10) KiB`。
kbg 写入 `L1[288,352) KiB`，vb 写入 `L1[352,480) KiB`；V=256 时单 task vb 为 32 KiB。
MTE2/V/MTE3 事件闭合后按 task 发布 Stage6 ready。

### Stage7：Cube，W/U

原路径依次计算 `W=A@kbg`、`U=A@vb`，L0A/L0B 的 S7 bank 为 `[32,48)` 和 `[48,64) KiB`，
L0C W/U 为 `[0,64)` 与 `[64,128) KiB`，各保留一个 `64×128 FP32` 结果。W/U 以 BF16
ROW_MAJOR Fixpipe 到 UB：`W[172,204) KiB`、`U[204,236) KiB`，下一 pack 由所属 AIV drain
到 GM；V=256 的 U 两个 128 列半块直接写 GM，不走 UB retained。

本轮已在 `K=V=128` 路径实现 packed W/U：L0A 只装载一次 A，kbg/vb 分别装载到同一
packed-NZ L0B 的 N 方向 `[0,128)` 与 `[128,256)` 半区，一次
`C[64,256]FP32=A[64,64]BF16 @ B[64,256]BF16`。 packed L0B 为 32 KiB，L0C 为 64 KiB，
均位于现有 S7 bank 内。两个输出从 L0C 的 `(0,0)`、`(0,128)` 分别取出，以 BF16
ROW_MAJOR Fixpipe 到原 W/U UB；只写 `chunk.M` 个有效行，下一 pack 仍由所属 AIV drain
到 GM。L1 输入与 L0 tile 均采用仓内 KDA 同类实现验证过的 TLA `GetTile` 形式，不把两个
独立 128 列输出误当成两个独立 L0C 布局。

V=256 保持原三次 MMAD 路径。packed 路径沿用 event 0 的 M_MTE1／MTE1_M／M_FIX／FIX_M
生命周期：下一 task 覆写 L0A/L0B 前等待上一 task 的 M_MTE1，MMAD 前等待上一 L0C
Fixpipe 完成，W/U 两次取出后再发布 FIX_M。跨核 drain 协议不变。当前状态是“实现与静态
复核完成，待 CANN 构建、精度和性能验证”，尚不能据此声明数值正确或性能获益。

### 搬运与同步汇总

Stage3 保持原始全量 Load、Mul 和 Store；Stage7 目标减少一次 A 的 L1→L0A 装载和一次
MMAD launch，不增加 GM 流量或 workspace。所有输入仍按原路径从 GM 读取，L2 策略不变。
跨核 Stage1→2、Stage2→3、Stage3→4、Stage4→下一 pack Stage1、Stage6→7、Stage7→drain
的 event id、方向、生产 pipe 和计数均保持不变；核内只复用已等待完成的 HardEvent，
不提前释放仍被 MMAD/Fixpipe 使用的 L0/L1/UB。

开发期实测、精度和逐轮采用结论见 [validation.md](validation.md)。
