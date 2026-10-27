# RecurrentGatedDeltaRule 原始流程与 timer 观测设计

基线 main `c9e2f1db`，A5/ascend950，AIV_ONLY。沿用默认 perf case 0：
B=T=Nk=Nv=1、K=V=128、g-only、BF16 state。实验 A/B 均不带入。
原接口和 CPU 标杆见 [api.md](api.md)，阶段方案见 [design.md](design.md)。

## 原始流程（未增加计时）

```text
stable Python / 固定 / host，取得当前 stream
  → stable adapter / 固定 / host，ACL descriptors + out + workspace
  → ACLNN L2 / 固定 / host，Contiguous / CreateView
      → L0 / 固定 / host，内部 out + ADD_TO_LAUNCHER_LIST_AICORE
      → ViewCopy(outRet,out) / 固定 / host，实际是否独立 task 以原 ATK profile 为准
  → tiling / 固定 / host，原 UB预算、vStep、dS/dO、dtype key、blockDim
  → kernel / 固定 / AIV，TPipe + RGDR 构造 + Init
      → ComputeAvgload / 固定 / AIV，扫描长度
      → batch 循环 / 动态 / AIV，跳过空序列、校验长度/accepted/index
          → CopyInGamaBeta / 动态 batch / AIV
              beta copy → DeQue(输入 ready 锚点) → FP32 Cast
              g copy → DeQue(输入 ready 锚点) → Exp
          → head 循环 / 动态 batch,head / AIV，IsCurrentBlock 判断归属
              → CopyInQKV / 动态 head / AIV
                  gk 初始化 → 原 V_MTE2 Set/Wait → copy → DeQue → Exp → PIPE_V
                  q/k/v copy → 各 DeQue → Cast → 原 PIPE_V → query scale
              → PrefetchState / 动态 head,V片 / AIV，MTE2 issue，EnQue
              → V片循环 / 动态 head,V片 / AIV
                  LoadPrefetchedState：DeQue(ready 锚点) → Cast/Copy → Free
                  若有下一 V片则预取
                  → token 循环 / 动态 head,V片,token / AIV
                      读取 state index/g/beta → 分配 state/attn 输出 slot
                      → 单次 VF（原两遍行循环，不拆分、不重排）
                          第一遍：g/gk 衰减 → Sbar·k reduction → dot/state 落 UB
                          原 LocalMemBar
                          第二遍：delta → rank-one 更新 → Snew·q reduction
                                    → state FP32 scratch/BF16快照 → out scalar
                          原 LocalMemBar → out cast/打包
                      → EnQue 输出
                      → 当前或上一个 pending 输出：DeQue(ready 锚点) → MTE3 issue
                  → 片末排空 pending 输出（不删除原等待/Free）
              → head 结束释放 gk
      → 原对象析构/退出
```

这是 recurrent decode，不包含 prefill 的 KKT/solve/WY/Cube/H/O 子 kernel。
输入队列 DeQue 仅保证对应输入 ready；输出队列 DeQue 仅保证供 MTE3 的向量数据 ready，
不保证随后 MTE3 已完成。裸 VF 调用周围的 scalar 时间戳仅视为 issue window。

## 观测点（在上图各同名节点前后插入）

固定点均为 AIV、本 kernel 一次：

| 原图节点 | FixedTiming | Tik/Tok | 含义 |
| --- | --- | --- | --- |
| kernel Init 至 Process 返回及末尾 drain | KERNEL_TIMING_IDX | TikNoBarrier / Tok<Overwrite> | 唯一新增的诊断末尾 PIPE_ALL drain；不含 timer 初始化/发布与析构 |
| TPipe/RGDR 构造及 Init | RGDR_INIT | TikNoBarrier / TokNoBarrier<Overwrite> | 初始化/指令下发窗口 |
| ComputeAvgload | RGDR_AVGLOAD | 同上 | 标量长度扫描 |
| Process | RGDR_PROCESS_ISSUE | 同上 | 含原有等待的整个工作下发窗口，嵌套所有动态点 |

动态点使用各类型在本核上的连续调用序号（不是全局 token id），每类型前 16 次记录；
metadata 保留每核每类型真实调用次数，超过容量明确标记截断。每槽单次写入，全部用
`TikNoBarrier(type,slot)` / `TokNoBarrier<Overwrite>(type,slot)`；不使用 Accumulate。

| 原图关键步骤 | DynamicTimingType（按枚举顺序） | 锚点 / 解释 |
| --- | --- | --- |
| 整个 batch gate 准备 | BATCH_GATES_ISSUE | 包含 copy、等待、Cast/Exp；有嵌套项，不能相加 |
| 整个 QKV/gk 准备 | HEAD_QKV_ISSUE | 同上 |
| gk DeQue | GK_COPY_WAIT | 输入 ready 等待 |
| gk Exp | GK_EXP_ISSUE | Exp 下发，保留后续原 PIPE_V |
| q DeQue | Q_COPY_WAIT | 输入 ready 等待 |
| k DeQue | K_COPY_WAIT | 输入 ready 等待 |
| v DeQue | V_COPY_WAIT | 输入 ready 等待 |
| QKV FP32 Cast | QKV_CAST_ISSUE | Cast 下发，保留后续原 PIPE_V |
| query scale | QUERY_SCALE_ISSUE | Muls 下发 |
| state 预取 | STATE_PREFETCH_ISSUE | DataCopyPad/EnQue 下发，不是完整 MTE2 耗时 |
| state 输入 DeQue | STATE_DEQUEUE_WAIT | 输入 ready 等待 |
| state FP32 Cast/Copy | STATE_CAST_COPY_ISSUE | 下发，不新增完成同步 |
| token metadata / g / beta | TOKEN_PARAMS | 原 GetValue 和地址计算 |
| state/attn 输出 AllocTensor | OUTPUT_ALLOC_WAIT | 可能含 queue slot 复用等待 |
| g/gk、dot、delta、rank-one、projection、cast | RECURRENT_VF_ISSUE | 单次融合 VF 共用观测窗口；这些步骤在 VF 内没有独立 scalar 完成边界，不插入时钟/屏障拆开它们 |
| 输出 EnQue | OUTPUT_ENQUEUE | 原队列发布下发 |
| attention 输出 DeQue | ATTN_DEQUEUE_WAIT | 数据 ready 等待；可能承接之前 VF |
| attention DataCopyPad/Free | ATTN_COPY_ISSUE | MTE3 下发和队列释放 |
| state 输出 DeQue | STATE_OUT_DEQUEUE_WAIT | 数据 ready 等待；可能承接之前 VF |
| state DataCopyPad/Free | STATE_OUT_COPY_ISSUE | MTE3 下发和队列释放 |
| beta DeQue | BETA_DEQUEUE_WAIT | 输入 ready 等待 |
| beta Cast | BETA_CAST_ISSUE | 下发 |
| g DeQue | G_DEQUEUE_WAIT | 输入 ready 等待 |
| g Exp | G_EXP_ISSUE | 下发 |
| 每个 ProcessHead | HEAD_TOTAL_ISSUE | 动态 head 外窗 |
| 每个 V片 | SLICE_TOTAL_ISSUE | 动态 V片外窗；包含片末 pending 排空 |
| 每个 token | TOKEN_TOTAL_ISSUE | 动态 token 外窗 |

数学关键步骤都位于 RECURRENT_VF_ISSUE 对应的原 VF，未伪造分项纯计算耗时。
只有管线下发和自然等待可在不改 VF 的条件下安全观测。若 VF/队列 ready 窗口显示主要等待，
后续再用原 ATK 系统 profile 的指令数据分析归约依赖；本轮不拆 VF 来制造测量边界。

## host、槽位与发布

普通 Python/ACLNN 签名保持不变，内部 optional timer 输入为空。独立 Profile C 接口及
私有 stable dispatcher 入口接受 caller 创建的 contiguous/int64/128B 对齐 timer；
校验最小长度使用同一布局常量，由 Profile size 查询函数供 stable 层读取。
Profile 仍调用相同 L0、相同 tiling 和同一个 kernel，保留 Contiguous/ViewCopy。

timer 为独立 GM 张量，不占原 UB/workspace，不与 state/out 别名。AIV_ONLY 使用
linear core_id=GetBlockIdx，容量 128 核；超过容量由 host 拒绝诊断调用。原数据无跨核同步，
本轮不新增 SyncAll。每核行与计数区 128B 对齐，扩展 metadata 行也独占 cache line。
所有动态类型真实次数由本核写入自己的 metadata；公共头部只由 block 0 写。
总窗口结束后各核调用一次 Publish，发布本核时间戳和计数；metadata 独立逐行发布。
显式发布使用 SINGLE_CACHE_LINE DCCI，不刷新整个 DCache。

无 timer 的常规路径使用 `RGDR<...,false>`，不构造计时上下文；诊断使用 `RGDR<...,true>`。
`ENABLE_TIMER=0` 删除打点，诊断采集必须因 raw 缺少记录而失败，不能报告成功。
早退位于 Process 内，统一返回入口后总窗口收尾和发布；空工作核仍有固定窗口，动态计数为 0。
