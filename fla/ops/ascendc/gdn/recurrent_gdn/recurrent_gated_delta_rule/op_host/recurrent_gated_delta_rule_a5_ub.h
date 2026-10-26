/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef RECURRENT_GATED_DELTA_RULE_A5_UB_H
#define RECURRENT_GATED_DELTA_RULE_A5_UB_H

#include <array>
#include <cstdint>
#include "../op_kernel/recurrent_gated_delta_rule_struct.h"

namespace RecurrentGatedDeltaRule {

constexpr uint64_t A5_UB_RESERVE_BYTES = 128;

struct A5UbProfile {
    uint32_t stateOutBufferNum = 1;
    uint32_t attnOutBufferNum = 1;
    uint32_t vStep = 0;
    uint32_t repeatTime = 0;
    uint64_t queueBytes = 0;
    uint64_t tmpBytes = 0;
    uint64_t totalBytes = 0;
    bool valid = false;
};

inline uint64_t AlignA5UbBytes(uint64_t bytes)
{
    constexpr uint64_t BLOCK_BYTES = 32;
    return (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

// Keep this ledger paired with arch35/RGDR::InitLocalBuffers. All queue slots
// are 32B aligned; both betaInUb and gamaInUb exist even when g is absent.
inline A5UbProfile CalcA5UbProfile(const RecurrentGatedDeltaRuleTilingData &tiling, uint32_t stateBytes,
                                  uint32_t stateOutBuffers, uint32_t attnOutBuffers, uint32_t rows)
{
    constexpr uint64_t TOKENS = 8;
    constexpr uint64_t INPUT_BYTES = 2;
    constexpr uint64_t FP32_BYTES = 4;
    const uint64_t alignK = (static_cast<uint64_t>(tiling.dk) + 15) / 16 * 16;
    const uint64_t alignV = (static_cast<uint64_t>(tiling.dv) + 15) / 16 * 16;
    const uint64_t gateElements = (TOKENS * tiling.nv + 15) / 16 * 16;
    A5UbProfile profile;
    profile.stateOutBufferNum = stateOutBuffers;
    profile.attnOutBufferNum = attnOutBuffers;
    profile.vStep = rows;
    profile.repeatTime = rows == 0 ? 0 : (tiling.dv + rows - 1) / rows;

    profile.queueBytes = 2 * AlignA5UbBytes(TOKENS * alignK * INPUT_BYTES); // q/k
    profile.queueBytes += AlignA5UbBytes(TOKENS * alignV * INPUT_BYTES); // v
    profile.queueBytes += AlignA5UbBytes(alignK * rows * stateBytes); // state input
    if (tiling.hasGama != 0) {
        profile.queueBytes += AlignA5UbBytes(TOKENS * tiling.nv * FP32_BYTES);
    }
    if (tiling.hasGamaK != 0) {
        profile.queueBytes += AlignA5UbBytes(TOKENS * alignK * FP32_BYTES);
    }
    profile.queueBytes += AlignA5UbBytes(TOKENS * tiling.nv * INPUT_BYTES); // beta
    profile.queueBytes += stateOutBuffers * AlignA5UbBytes(alignK * rows * stateBytes);
    profile.queueBytes += attnOutBuffers * AlignA5UbBytes(rows * INPUT_BYTES);

    profile.tmpBytes = rows * FP32_BYTES; // attnInUb
    profile.tmpBytes += TOKENS * (alignV + 2 * alignK) * FP32_BYTES; // v/q/k
    profile.tmpBytes += alignK * rows * FP32_BYTES; // FP32 recurrent state
    profile.tmpBytes += 2 * gateElements * FP32_BYTES; // betaInUb and gamaInUb
    profile.tmpBytes = AlignA5UbBytes(profile.tmpBytes);
    profile.totalBytes = profile.queueBytes + profile.tmpBytes + A5_UB_RESERVE_BYTES;
    return profile;
}

inline A5UbProfile SelectA5UbProfile(const RecurrentGatedDeltaRuleTilingData &tiling, uint32_t stateBytes,
                                    uint64_t ubBytes)
{
    A5UbProfile selected;
    if (tiling.t == 0 || tiling.dk != 128 || tiling.dv != 128 || tiling.nv == 0 || tiling.nv > 256 ||
        (stateBytes != 2 && stateBytes != 4)) {
        return selected;
    }
    const std::array<std::array<uint32_t, 2>, 3> candidates = {{{1, 1}, {1, 2}, {2, 2}}};
    for (const auto &buffers : candidates) {
        // T is a safe upper bound on every sequence length, including prefix
        // and empty sequences. T == B alone does not establish single-token decode.
        if (tiling.t == 1 && (buffers[0] != 1 || buffers[1] != 1)) {
            continue;
        }
        for (uint32_t rows = tiling.dv; rows >= 8; rows -= 8) {
            A5UbProfile candidate = CalcA5UbProfile(tiling, stateBytes, buffers[0], buffers[1], rows);
            if (candidate.totalBytes > ubBytes) {
                continue;
            }
            // Keep the original balanced-slice rule once the minimum number
            // of slices is known, rather than leaving a large slice plus a tiny tail.
            const uint32_t balancedRows = ((tiling.dv + candidate.repeatTime - 1) / candidate.repeatTime + 7) / 8 * 8;
            candidate = CalcA5UbProfile(tiling, stateBytes, buffers[0], buffers[1], balancedRows);
            candidate.valid = true;
            const uint32_t depth = candidate.stateOutBufferNum + candidate.attnOutBufferNum;
            const uint32_t selectedDepth = selected.stateOutBufferNum + selected.attnOutBufferNum;
            if (!selected.valid || candidate.repeatTime < selected.repeatTime ||
                (candidate.repeatTime == selected.repeatTime &&
                 (depth > selectedDepth || (depth == selectedDepth && candidate.vStep > selected.vStep)))) {
                selected = candidate;
            }
            break; // This profile's minimum slice count, with balanced rows.
        }
    }
    return selected;
}

} // namespace RecurrentGatedDeltaRule
#endif // RECURRENT_GATED_DELTA_RULE_A5_UB_H
