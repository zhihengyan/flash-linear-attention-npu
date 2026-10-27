#ifndef ASCEND_TIMER_COMMON_H
#define ASCEND_TIMER_COMMON_H

#include <stdint.h>

namespace AscendTimer
{
// RGDR AIV_ONLY layout. See docs/timer_flow.md; all names match the parser.
// 固定计时项（整个 kernel 只跑一次）
enum FixedTiming : int32_t
{
    KERNEL_TIMING_IDX,
    RGDR_INIT,
    RGDR_AVGLOAD,
    RGDR_PROCESS_ISSUE,
    FIXED_TIMING_COUNT
};

// 动态计时项类型（循环内按 iter 重复；常见构成：总时长 / WAIT 等待段 / 细粒度子段）
enum DynamicTimingType : int32_t
{
    BATCH_GATES_ISSUE,
    HEAD_QKV_ISSUE,
    GK_COPY_WAIT,
    GK_EXP_ISSUE,
    Q_COPY_WAIT,
    K_COPY_WAIT,
    V_COPY_WAIT,
    QKV_CAST_ISSUE,
    QUERY_SCALE_ISSUE,
    STATE_PREFETCH_ISSUE,
    STATE_DEQUEUE_WAIT,
    STATE_CAST_COPY_ISSUE,
    TOKEN_PARAMS,
    OUTPUT_ALLOC_WAIT,
    RECURRENT_VF_ISSUE,
    OUTPUT_ENQUEUE,
    ATTN_DEQUEUE_WAIT,
    ATTN_COPY_ISSUE,
    STATE_OUT_DEQUEUE_WAIT,
    STATE_OUT_COPY_ISSUE,
    BETA_DEQUEUE_WAIT,
    BETA_CAST_ISSUE,
    G_DEQUEUE_WAIT,
    G_EXP_ISSUE,
    HEAD_TOTAL_ISSUE,
    SLICE_TOTAL_ISSUE,
    TOKEN_TOTAL_ISSUE,
    DYNAMIC_TYPE_COUNT_ENUM
};

// ========== 2. 公共常量定义（CPU/NPU共用） ==========
// 所有动态项默认最大迭代次数
static constexpr int MAX_DYNAMIC_ITER = 16;

// CPU端：对齐相关常量（128字节(16元素)对齐，避免多核写一个cache line）
// 每个计时项占 2 个 int64 槽位: [start_cycle, end_cycle]
static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;                                     // 每核原始计数器数
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;                // 16倍对齐后每核计数器数 128B对齐 每个核写入N_TIMING_COUNTER_PER_CORE_ALIGN个数
static constexpr int N_CORE_COUNT = 128; // AIV_ONLY storage capacity, not a hardware core-count assumption.
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;                           // 总基础计数器数

// 动态迭代次数存储区对齐配置（每核动态项存储区16元素对齐）
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;                         // 动态项类型总数
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;    // 每核动态迭代次数区16对齐 128B对齐
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT; // 动态迭代次数总存储区大小，每个核cacheline不重叠
// RGDR diagnostic metadata; each core owns an independent 512B row.
static constexpr int BASE_BUFFER_SIZE = N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE;
static constexpr int META_WORDS_PER_CORE = 64;
static constexpr int META_OFFSET = BASE_BUFFER_SIZE;
static constexpr int HEADER_OFFSET = META_OFFSET + N_CORE_COUNT * META_WORDS_PER_CORE;
static constexpr int HEADER_WORDS = 16;
static constexpr int TOTAL_BUFFER_SIZE = HEADER_OFFSET + HEADER_WORDS;
static constexpr int64_t RGDR_TIMER_MAGIC = 0x5247445254494D45LL;
static constexpr int64_t RGDR_TIMER_VERSION = 1;

}
// ========== 4. 计时开关宏 ==========
#ifndef ENABLE_TIMER
#define ENABLE_TIMER 1 // 构建可覆盖为0；所有计时调用均用TIMER_BLOCK。
#endif
#ifndef ASCEND_TIMER_EXPLICIT_GM_PUBLISH
#define ASCEND_TIMER_EXPLICIT_GM_PUBLISH 1 // 仅确认目标发布需求/API/对齐后启用。
#endif
#if ENABLE_TIMER
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
        code ;            \
    } while (0);
#else
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
    } while (0);
#endif

#endif // ASCEND_TIMER_COMMON_H
