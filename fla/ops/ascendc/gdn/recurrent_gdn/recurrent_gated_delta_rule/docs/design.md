# RecurrentGatedDeltaRule：A5 初始 BF16 state Cast 融合实验

方案设计规则版本：`V2`。本轮从已保留的 `fec1e18b`（`codex/gdn-a5-tail-scratch-elision`）创建，
实验分支为 `codex/gdn-a5-state-cast-fusion`。历史 main 为 `c9e2f1db`。
接口见 [api.md](api.md)，开发记录和待验证项见 [validation.md](validation.md)。

## 1. 目标

仅优化用户固定的 A5 decode case：B=T=Nk=Nv=1、K=V=128、BF16 state、g-only。
用户回传 scratch 的 200 条精度全过；最近四轮 RGDR Task Duration 均值为
4.49/4.49/4.43/4.46 μs，等权汇总 4.4675 μs；对应端到端均值为 18.3365 μs。
VF 系数读取及显式 V_S 删除实验未观测到稳定收益，本轮不继承它们，原系数标量读取与 V_S 保留。
本轮只改变初始 state 的转换与消费位置：BF16 输入队列直接供主 VF 第一遍加载，在寄存器中
完成精确扩展。第一遍仍写 gated FP32 scratch，第二遍仍读它；两遍循环、归约树和输出保持。
队列 slot 必须保留到 VF 最后一次读取之后再 Free，不能沿用独立 Cast 之后立即 Free 的时机。
性能口径固定为原 ATK `op_summary` 中 `OP Type=RecurrentGatedDeltaRule` 的 `Task Duration(us)`，
其它 Neg/有限性检查 task 不混入算子时间。沿用原输入、预热、每轮 20 样本及统计方式，
以 fec1e18b/本轮分支交错复测，main 在最终端到端目标评估时同条件复测。
目标是在跨轮波动下仍能降低 RGDR 均值/中位数，且精度、其它路径不退化；未实测前不宣称收益。
目标 kernel 是默认 case 实际命中的 RecurrentGatedDeltaRule BF16 single-work 实例。
按历史 main 18.83 μs 估算，累计 5% 端到端目标为 17.8885 μs，需在 scratch 上再节省
约 0.448 μs；若其它开销不变且完全传递，对应 kernel 约 4.02 μs。该换算仅用于可达性评审，
不是同轮实测目标已达成，也不保证 kernel 节省能等量传递。预热、环境、20 样本和统计沿用原 ATK，
实际预热次数与频率状态从内网日志归档，不自行补造数值。

可达性：源码层面省去输入 Cast 的一次 FP32 UB 写入和 VF 第一遍对应读取，各 64 KiB，
共 128 KiB/work；BF16 原始读取 32 KiB 和 GM 读写不变。若原编译器已消除往返，或访存不在
关键路径，收益可能为零；逐行增加的 Cast、寄存器压力与队列等待也可能抵消收益。
用户提供的 CANN 9.1 符号表保留独立 Cast VF 和主 VF，但指令均显示 `<not available>`，
不能证明实际保留的搬运量。用户已明确授权以单变量实验验证该假设；没有可信的时间收益下界。

## 2. Stage 0：Vector，保持原递推，特化单 work 控制

### 任务、模板与合法性

主 Stage 仍由一个 AIV 执行整个 value head，不引入 Cube、跨核归约或跨核 flag。
模板参数增加 `SINGLE_WORK`，列出 BF16/general、FP32/general、BF16/single 三个合法组合。
host 仅在 A5、BF16、B=T=Nk=Nv=1、K=V=vStep=128、blockDim=1、stateStride2=128、hasG=1、hasGK=0、
hasAccepted=0 时设置 SINGLE_WORK；其余选择 general。tiling struct 不增加字段。
A2/A3 仍使用原类和 general 参数，快速拉起的 TORCH_MODE struct 布局也保持。

special 实例将相关 shape/gate 成员初始化为这些已证明的常量，使原有小函数可以常量传播；
general 实例的成员初始化、调度、原 LoadPrefetchedState 和 VF 数学保持。

VF 新增内部编译期参数 `writeUpdatedScratch=true`；Compute 将 `!singleWork` 传给它。
因此仅已有 SINGLE_WORK=1 实例使用 false，其余 BF16/FP32、单/多 token 通用实例均使用 true。
它不进入公开 kernel 模板、host TilingKey 或 tiling struct，不扩大原专用路径范围。

新增 VF 内部参数 `inputStateBf16=false` 和末尾 UB 输入指针；只有 singleWork 的固定 g-only
调用使用 true，其余四种 general 调用使用默认 false 并传 nullptr。Compute 的私有指针参数
默认 nullptr，不改变 general 的原三参数调用。静态断言限制 true 为 BF16、g-only、免写 updated
scratch 的实例。通用 VF 在编译期移除新分支；其生成代码与额外未使用参数仍需 CANN 验证。

### 设备侧控制与地址

1. 读取 `cuSeqlens[0]` 与 `cuSeqlens[1]`。在 T=1 下，仅 prefix=0、length=1 能产生有效工作；
   合法空工作 prefix=1、length=0 直接返回，不能根据 T=B 就绕过这一步。
2. 读取一次 `ssmStateIndices[0]`，复用作初始 state 与最终 state 的 block index。
   地址为 `stateStride0 * index`，head=0、V片起点=0；不假定 block index 为零或 block stride 连续。
3. 原 `CopyInGamaBeta(0,1)` → `CopyInQKV(0,0,1)` → `PrefetchState(stateOffset,128)` →
   stateInQueue 的原 `DeQue<stateType>()` → 一对显式 `V_S` Set/Wait → 原 g/beta Scalar GetValue。
   专用路径绕过 LoadPrefetchedState 的整体 Cast/Free，将已 ready 的 stateLocal 裸 UB 地址
   传入 Compute。VF 使用期间 stateLocal 仍持有原 slot 0；Compute 返回后调用一次
   stateInQueue.FreeTensor(stateLocal)，然后进入原输出流程。
4. 原 `CopyOutAttn(0,128)` → `CopyOutState(stateOffset,128)`。单 token、单分片下，原 host 可选的
   `(state,attn)` 深度 `(1,1)/(1,2)/(2,2)` 均按此顺序写回；特化只移除 pending 状态变量和循环控制，
   不改变队列深度。

相较 main，移除 ComputeAvgload、IsCurrentBlock、batch/head/V/token 四层单次循环及通用偏移计算。
GM 元数据 GetValue 从 5 次降为 3 次：长度不再为平均负载重复读取，state index 不再为输出重复读取。
这只是逻辑请求数，不等价于实际减少同样次数的 HBM 访问。

### 数值、数据和资源

保持原两遍 VF、归约树、gate Exp、q scale、输出 Cast 舍入和全部 GM 输出。
数学仍为 FP32 gate/state-dot → delta → rank-one 更新 → state-query projection → BF16 state/out。
R21：CPU 标杆 `state.to(float32) → exp(g) * state → dot → delta → update → output` 对应
原整体 Cast 改为每行 BF16 load 后两次 Reg Cast，其后运算完全沿用。有限 BF16 的符号、8 位
指数和 7 位尾数均可由 FP32 精确表示，数学上不引入额外舍入或扩大值域；保留原控制状态、
mask、Exp 和输出舍入，不引入 FMA。设备对 subnormal 的处理及真实指令仍需原精度标准验证。

原 BF16 路径的 FP32 DIST_DINTLV_B32 按偶/奇 K 索引分入 state0/state1。新路径以
DataCopy 将完整 128 个 BF16（256 B）读入一只寄存器，RegLayout::ZERO/ONE 分别扩展
偶/奇位置为两个 64-lane FP32 寄存器。Cast 使用 UNKNOWN saturation/rounding 和 ZEROING，
mask 仍为 fullFp32Mask。因此与原 key0/key1、query0/query1 及后续配对 Add/ReduceSum 一致，
不能改成 K 前半/后半布局。每行地址为 initialState + row*kLength，目标固定 K=128，
首地址按原队列对齐、步长 256 B；128 行最后一次读结束在原 32768 B 输入区域边界。
API 语义依据：[RegLayout](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00237.html)、
[Cast](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0398.html)
及 [DIST_DINTLV_B32](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/LoadAlign_continuous.md)。
这些资料用于 layout/trait 选择，不替代目标 CANN 9.1 编译和 A5 验证。

FP32 state scratch 的生命周期如下，地址仍为 tmpBuff 相对区间 `[12800,78336)`：

| 操作 | fec1e18b | 本轮 SINGLE_WORK=1 | 最后消费者 |
| --- | ---: | ---: | --- |
| 输入 BF16 state 转换后写 FP32 | 64 KiB | 0 | 本轮直接在寄存器消费 |
| VF 第一遍从 FP32 scratch 读初始 state | 64 KiB | 0 | 改为读取原 BF16 输入 32 KiB |
| VF 第一遍写 gated state | 64 KiB | 64 KiB | VF 第二遍加载，必须保留 |
| VF 第二遍写 updated state | 0 | 0 | 沿用尾部免写回 |
| BF16 state GM 输出 | 32 KiB | 32 KiB | 调用方，原输出不变 |

第二遍的 attention 投影和 stateOut 转换直接消费更新后的 FP32 寄存器，不回读 updated scratch。
目标路径只有一个 token、一个 V 分片，之后只写出 GM 并退出；scratch 保持 gated state 不影响输出。
general 路径的下一个 token 会读取 updated scratch，所以 `writeUpdatedScratch=true` 必须保持原行为。
两个 LocalMemBar 仍保留，分别保护第一遍 scratch 与 attention 临时结果的后续读取。

L1/L0/算法 GM workspace 均为 0；保留原 16 MiB 系统 workspace。UB 使用原 InitLocalBuffers：

| 对象 | 目标 case 的逻辑字节数 | 生命周期 |
| --- | ---: | --- |
| q/k/v 输入队列 | 各 2048 | MTE2 → Cast → Free |
| state 输入队列 | 32768 | MTE2 → DeQue ready → VF 第一遍末次读取 → Compute 返回后 Free |
| g / beta 输入队列 | 32 / 16 | MTE2 → Exp/Cast → Free |
| state 输出队列，两 slot | 65536 | VF → MTE3 → Free |
| attention 输出队列，两 slot | 512 | VF → MTE3 → Free |
| tmpBuff 中实际命名 scratch | 78464 | 下述原地址保持 |

以上未加入 TPipe 对齐/管理开销，可用 UB 继续由原 platform 查询和 tiling 预算决定。
tmpBuff 相对字节区间保持：attn `[0,512)`、v `[512,4608)`、q `[4608,8704)`、k `[8704,12800)`、
FP32 state `[12800,78336)`、beta `[78336,78400)`、g `[78400,78464)`。
所有 TQue 物理 slot 地址由原 TPipe 分配，不新增别名、存储份数或缓冲回绕。
输入队列与 tmpBuff 的 FP32 state 是原本就独立的存储；只是将输入 slot 的持有时间延长，
不会把输入指针重解释为 scratch，也不会让输出覆盖仍在读取的输入。FP32 state 区域从
VF 第一遍逐行 gated Store 后才有有效内容，第一遍严禁从其中读取未初始化数据。
无额外 UB 分配；新增一只逻辑 BF16 寄存器仅在每行两次 Cast 前后存活，峰值寄存器和 spill 待实测。

每次有效工作逻辑读取 BF16 state 32 KiB、写回 32 KiB，Q/K/V 共 768 B、g/beta 共 6 B，均不变。
新路径复用的 index 位于 scalar 局部变量，不占 UB，不改变任何输入/输出地址所有权。

### 同步与退出

保留所有 Copy helper、general LoadPrefetchedState、PIPE_V 和两处 VF LocalMemBar。
专用 state 的 DeQue 保留在原输入 ready 位置，Free 从整体 Cast 后延至主 VF 消费后。
输入 ready 和输出 ready 继续由原队列事件保证。专用路径在读取 `gamaInUb/betaInUb` 之前，
通过 TPipe 动态取得一个 `HardEvent::V_S` event ID，先 SetFlag 再 WaitFlag，形成一次完成依赖。
生产方是此前发射的 gate Exp、beta Cast、Q/K/V Cast/scale，消费方是标量读取及
后续 Compute；每次有效调用只执行一对 Set/Wait，不持有未消费的通知，不引入跨核同步或 PIPE_ALL。
输出仍以原 CopyOut 函数释放，TPipe 的退出生命周期不变。空工作在读取/写入 state、分配输出前返回。
空工作也在取得该 event ID 之前返回，因此不会出现缺少 Set 的 Wait。

| 资源 / slot | 生产、等待与消费闭环 | 本轮数量 / 复用 |
| --- | --- | --- |
| stateInQueue slot 0，32 KiB BF16 | 原 MTE2 Copy/EnQue → ProcessSingleWork DeQue → VF load/Cast → Compute 返回 → Free | 原 1 slot；本 work 无后续输入复用 |
| g/beta、QKV 输入队列 | 原搬入/EnQue/DeQue → 原 Exp/Cast/scale → 原 Free | 不变 |
| 显式 V_S | Fetch → Set → Wait → Scalar 读 g/beta → Compute | 原一对；有效工作一次，空工作零次 |
| gated scratch 与 dot tmp | VF 第一遍 Store → 原 LocalMemBar → VF 第二遍 Load | 原一处；不删除 |
| attention tmp | VF 第二遍 Store → 原 LocalMemBar → 输出转换 | 原一处；不删除 |
| 输出队列、MTE3 | 原 Compute EnQue → 原 CopyOut DeQue/写出/Free | 原深度与释放闭环 |

同步伪代码：检查 prefix/length → 读 index → gate/beta 和 QKV 准备 → PrefetchState(slot0)
→ DeQue(slot0 ready) → 原 V_S Set/Wait → Scalar g/beta → Compute(原输出 Alloc，VF 读原 BF16
slot0、原两遍计算和屏障、输出 EnQue) → Free(slot0) → CopyOutAttn → CopyOutState → 原 TPipe 退出。
有效单 work 无第二次 state Alloc，不引入 slot 回绕；空工作不发射 state copy/DeQue/Free/VF/event。

2026-10-29 用户回传：无显式同步时 case 0 的 out/stateRef 全部非有限；强制 general 后通过；
恢复专用路径并增加上述 V_S 后通过。这支持前置 Vector 完成时序相关问题；由于该事件覆盖所有
前置 Vector 操作，尚不能单独归因于 gate/beta 中的某一个读取，也未证明编译器漏插同步的具体原因。
修复前的性能数据不能作为本版本的验收结果，必须在精度回归后重新测量。

### 设计复核

R01/R20：本例仅一个 AIV/work，未改变原 head 所有权；其它工作全部回退。
R03/R08/R11/R17：沿用原 UB/queue/workspace 和地址，无新增容量或复用风险。
R10/R16：GM state 搬入/写回次数不变，只移除初始化 FP32 中转；原 gated scratch 仍复用。
R21：精确扩展提前到每行消费点、保持偶/奇 layout 和全部后续数学。本机无 CANN/NPU；
源码检查不替代本轮编译、精度、MSS、确定性或性能。case 188 和全部测试输入按用户要求不改。
