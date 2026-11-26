#ifndef ASCEND_TIMER_COMMON_H
#define ASCEND_TIMER_COMMON_H

#include <stdint.h>

namespace AscendTimer
{
enum FixedTiming : int32_t
{
    KERNEL_TIMING_IDX = 0,
    FIXED_TIMING_COUNT
};

// Dynamic slots are localPackSeq * TasksPerPack() + taskIdx. Events use Overwrite so their
// absolute timestamps can be compared across AIC/AIV. Instant events have
// adjacent Tik/Tok calls and represent a point, not a duration.
enum DynamicTimingType : int32_t
{
    AIV_S1_TASK,
    AIV_S1_WAIT_NEGL_FREE,
    AIV_S1_UPLOAD_K,
    AIV_S1_GATE_BETA,
    AIV_S3,
    AIV_S6,
    AIV_WU_DRAIN,
    AIV_S7_WAIT,
    AIC_S1_DIRECT_K_COPY,
    AIC_S2_WAIT_S1,
    AIC_S2_KKT,
    AIC_S3_WAIT,
    AIC_S4_PREFILL_I,
    AIC_S4_ACC,
    AIC_S4_NEGL_TO_L0B,
    AIC_S4_Y_DUMP,
    AIC_S4_FREE_SET,
    AIC_S5_WAIT_Y,
    AIC_S5_COMPUTE,
    AIC_S5_DUMP,
    AIC_S6_WAIT,
    AIC_S7,
    AIV_S3_WAIT_KKT,
    AIV_S0,
    AIV_S1_Q_NORM,
    AIV_S3_PREPARE_GATE,
    AIC_S7_WAIT_A,
    DYNAMIC_TYPE_COUNT_ENUM
};

// Covers the busiest fused c1 group (300 slots including final drain/wait).
static constexpr int MAX_DYNAMIC_ITER = 320;
static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;
static constexpr int N_CORE_COUNT = 96;
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT;
}

#ifndef PREPARE_TIMER_DIAGNOSTIC
#define PREPARE_TIMER_DIAGNOSTIC 0
#endif
#ifndef ENABLE_TIMER
#define ENABLE_TIMER PREPARE_TIMER_DIAGNOSTIC
#endif
#ifndef ASCEND_TIMER_EXPLICIT_GM_PUBLISH
#define ASCEND_TIMER_EXPLICIT_GM_PUBLISH 1
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
