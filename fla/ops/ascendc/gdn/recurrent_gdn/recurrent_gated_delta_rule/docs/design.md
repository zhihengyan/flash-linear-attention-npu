# RecurrentGatedDeltaRule：A5 单 work 标量路径实验

方案设计规则版本：`V2`。基线 main `c9e2f1db`，分支 `codex/gdn-a5-single-work-fastpath`。
接口见 [api.md](api.md)，开发记录和待验证项见 [validation.md](validation.md)。

## 1. 目标

仅优化用户固定的 A5 decode case：B=T=Nk=Nv=1、K=V=128、BF16 state、g-only。
用户交错复测的 main 主 kernel 均值为 4.87 μs；实验 A 约退化 9%，B 平均改善约 0.8% 且尚未稳定确认。
本轮从 main 独立实现，不混入 VF 融合、scratch 写回消除、A5 UB 重算、输出深度调整或 timer。

假设是单 work 的通用扫描、任务归属判断和重复元数据读取占有可减少的标量成本。
性能口径固定为原 ATK `op_summary` 中 `OP Type=RecurrentGatedDeltaRule` 的 `Task Duration(us)`，
其它 Neg/有限性检查 task 不混入算子时间。沿用原输入、预热、采样和环境，main/C 交错复测。
目标是在跨轮波动下仍能降低 RGDR 均值/中位数，且精度、其它路径不退化；未实测前不宣称收益。
由于没有本轮 main 的完整 PMU，不能按此前其它采集的 Scalar 比例预测固定加速幅度。

可达性：算法、VF 和 GM state 字节数不变，只减少启动控制和元数据请求，不以带宽峰值改进为前提。
这是一轮小范围可归因实验；若 Task Duration 没有稳定下降，则不继续堆叠该方案。

## 2. Stage 0：Vector，保持原递推，特化单 work 控制

### 任务、模板与合法性

主 Stage 仍由一个 AIV 执行整个 value head，不引入 Cube、跨核归约或跨核 flag。
模板参数增加 `SINGLE_WORK`，列出 BF16/general、FP32/general、BF16/single 三个合法组合。
host 仅在 A5、BF16、B=T=Nk=Nv=1、K=V=vStep=128、blockDim=1、stateStride2=128、hasG=1、hasGK=0、
hasAccepted=0 时设置 SINGLE_WORK；其余选择 general。tiling struct 不增加字段。
A2/A3 仍使用原类和 general 参数，快速拉起的 TORCH_MODE struct 布局也保持。

special 实例将相关 shape/gate 成员初始化为这些已证明的常量，使原有小函数可以常量传播；
general 实例的成员初始化、调度和原 VF 不变。

### 设备侧控制与地址

1. 读取 `cuSeqlens[0]` 与 `cuSeqlens[1]`。在 T=1 下，仅 prefix=0、length=1 能产生有效工作；
   合法空工作 prefix=1、length=0 直接返回，不能根据 T=B 就绕过这一步。
2. 读取一次 `ssmStateIndices[0]`，复用作初始 state 与最终 state 的 block index。
   地址为 `stateStride0 * index`，head=0、V片起点=0；不假定 block index 为零或 block stride 连续。
3. 原 `CopyInGamaBeta(0,1)` → `CopyInQKV(0,0,1)` → `PrefetchState(stateOffset,128)` →
   `LoadPrefetchedState(128)`；读取原 UB 中的 g/beta，调用原 `Compute(128,0,0)`。
4. 原 `CopyOutAttn(0,128)` → `CopyOutState(stateOffset,128)`。单 token、单分片下，原 host 可选的
   `(state,attn)` 深度 `(1,1)/(1,2)/(2,2)` 均按此顺序写回；特化只移除 pending 状态变量和循环控制，
   不改变队列深度。

相较 main，移除 ComputeAvgload、IsCurrentBlock、batch/head/V/token 四层单次循环及通用偏移计算。
GM 元数据 GetValue 从 5 次降为 3 次：长度不再为平均负载重复读取，state index 不再为输出重复读取。
这只是逻辑请求数，不等价于实际减少同样次数的 HBM 访问。

### 数值、数据和资源

保持原两遍 VF、归约树、gate Exp、q scale、Cast 舍入和最终 FP32 scratch 写回。
数学仍为 FP32 gate/state-dot → delta → rank-one 更新 → state-query projection → BF16 state/out。
R21：未做代数重排、FMA 替换或输入值域变化，不引入新的数值风险。

L1/L0/算法 GM workspace 均为 0；保留原 16 MiB 系统 workspace。UB 使用原 InitLocalBuffers：

| 对象 | 目标 case 的逻辑字节数 | 生命周期 |
| --- | ---: | --- |
| q/k/v 输入队列 | 各 2048 | MTE2 → Cast → Free |
| state 输入队列 | 32768 | MTE2 → Cast → Free |
| g / beta 输入队列 | 32 / 16 | MTE2 → Exp/Cast → Free |
| state 输出队列，两 slot | 65536 | VF → MTE3 → Free |
| attention 输出队列，两 slot | 512 | VF → MTE3 → Free |
| tmpBuff 中实际命名 scratch | 78464 | 下述原地址保持 |

以上未加入 TPipe 对齐/管理开销，可用 UB 继续由原 platform 查询和 tiling 预算决定。
tmpBuff 相对字节区间保持：attn `[0,512)`、v `[512,4608)`、q `[4608,8704)`、k `[8704,12800)`、
FP32 state `[12800,78336)`、beta `[78336,78400)`、g `[78400,78464)`。
所有 TQue 物理 slot 地址由原 TPipe 分配，不新增别名、存储份数或缓冲回绕。

每次有效工作逻辑读取 BF16 state 32 KiB、写回 32 KiB，Q/K/V 共 768 B、g/beta 共 6 B，均不变。
新路径复用的 index 位于 scalar 局部变量，不占 UB，不改变任何输入/输出地址所有权。

### 同步与退出

保留各 Copy/Compute 函数中的原 Alloc/EnQue/DeQue/Free、PIPE_V 和 VF LocalMemBar。
输入 ready 和输出 ready 继续由原队列事件保证；没有新 barrier、手写事件或跨核同步。
输出仍以原 CopyOut 函数释放，TPipe 的退出生命周期不变。空工作在读取/写入 state、分配输出前返回。

### 设计复核

R01/R20：本例仅一个 AIV/work，未改变原 head 所有权；其它工作全部回退。
R03/R08/R11/R17：沿用原 UB/queue/workspace 和地址，无新增容量或复用风险。
R10/R16：仅复用同一 token 的元数据，不更改 state 的搬入/写回次数。
R21：VF 数学、顺序和转换不变。用户已授权此方向的编码；无内网设备，真实编译、精度和性能均标为待验证。
