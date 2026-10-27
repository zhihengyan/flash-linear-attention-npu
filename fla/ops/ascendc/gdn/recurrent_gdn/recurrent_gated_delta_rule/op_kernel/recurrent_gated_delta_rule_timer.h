#ifndef RECURRENT_GATED_DELTA_RULE_TIMER_H
#define RECURRENT_GATED_DELTA_RULE_TIMER_H

#include "AscendTimerV2_device.hpp"
#include "recurrent_gated_delta_rule_struct.h"

// Diagnostic-only scalar state, external to the ordinary RGDR instantiation.
struct RgdrTimerContext {
    AscendTimerDevice all;
    AscendTimerDevice stage;
    AscendTimerDevice process;
    AscendTimerDevice dynamic;
    uint32_t calls[AscendTimer::DYNAMIC_TYPE_COUNT]{};

    __aicore__ inline void Init(GM_ADDR address)
    {
        all.Init(address);
        stage.Init(address);
        process.Init(address);
        dynamic.Init(address);
    }

    __aicore__ inline int Begin(AscendTimer::DynamicTimingType type)
    {
        const uint32_t slot = calls[static_cast<int>(type)]++;
        if (slot >= AscendTimer::MAX_DYNAMIC_ITER) {
            return -1;
        }
        dynamic.setDynamicActualIter(type, slot + 1);
        dynamic.TikNoBarrier(type, static_cast<int>(slot));
        return static_cast<int>(slot);
    }

    __aicore__ inline void End(AscendTimer::DynamicTimingType type, int slot)
    {
        if (slot >= 0) {
            dynamic.template TokNoBarrier<Overwrite>(type, slot);
        }
    }

    __aicore__ inline void Publish(const RecurrentGatedDeltaRule::RecurrentGatedDeltaRuleTilingData &t,
                                  uint32_t stateBytes)
    {
        using namespace AscendTimer;
        AscendC::GlobalTensor<int64_t> data;
        data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(all.GetTAddr()));
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t offset = META_OFFSET + core * META_WORDS_PER_CORE;
        for (int i = 0; i < DYNAMIC_TYPE_COUNT; ++i) {
            data.SetValue(offset + i, calls[i]);
        }
        data.SetValue(offset + 32, core);
        data.SetValue(offset + 33, AscendC::GetBlockNum());
        data.SetValue(offset + 63, RGDR_TIMER_MAGIC);
        all.Publish(); // All timer objects share this core's row; publish once.
        PublishLines(data, offset, META_WORDS_PER_CORE);
        if (core == 0) {
            const int64_t header[HEADER_WORDS] = {
                RGDR_TIMER_MAGIC, RGDR_TIMER_VERSION, TOTAL_BUFFER_SIZE, AscendC::GetBlockNum(),
                t.b, t.t, t.nk, t.nv, t.vStep, t.stateOutBufferNum, t.attnOutBufferNum,
                stateBytes, FIXED_TIMING_COUNT, DYNAMIC_TYPE_COUNT, MAX_DYNAMIC_ITER, ENABLE_TIMER};
            for (int i = 0; i < HEADER_WORDS; ++i) {
                data.SetValue(HEADER_OFFSET + i, header[i]);
            }
            PublishLines(data, HEADER_OFFSET, HEADER_WORDS);
        }
    }

    __aicore__ inline void PublishLines(AscendC::GlobalTensor<int64_t> &data, uint32_t offset, uint32_t words)
    {
#if ENABLE_TIMER && ASCEND_TIMER_EXPLICIT_GM_PUBLISH
        for (uint32_t word = 0; word < words; word += 64 / sizeof(int64_t)) {
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                            AscendC::DcciDst::CACHELINE_OUT>(data[offset + word]);
        }
#endif
    }
};
#endif
