/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule_struct.h
 * \brief Plain tiling struct shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECURRENT_GATED_DELTA_RULE_STRUCT_H
#define RECURRENT_GATED_DELTA_RULE_STRUCT_H

#include <cstdint>

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace RecurrentGatedDeltaRule {

#define RGDR_TPL_BF16 10
#define RGDR_TPL_FP32 30

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(RecurrentGatedDeltaRule,
    ASCENDC_TPL_DTYPE_DECL(D_T_STATE, RGDR_TPL_BF16, RGDR_TPL_FP32),
    ASCENDC_TPL_BOOL_DECL(SINGLE_WORK, 0, 1),
    ASCENDC_TPL_BOOL_DECL(STATIC_UB, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_STATE, RGDR_TPL_BF16),
        ASCENDC_TPL_BOOL_SEL(SINGLE_WORK, 0),
        ASCENDC_TPL_BOOL_SEL(STATIC_UB, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_STATE, RGDR_TPL_FP32),
        ASCENDC_TPL_BOOL_SEL(SINGLE_WORK, 0),
        ASCENDC_TPL_BOOL_SEL(STATIC_UB, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_STATE, RGDR_TPL_BF16),
        ASCENDC_TPL_BOOL_SEL(SINGLE_WORK, 1),
        ASCENDC_TPL_BOOL_SEL(STATIC_UB, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_STATE, RGDR_TPL_BF16),
        ASCENDC_TPL_BOOL_SEL(SINGLE_WORK, 0),
        ASCENDC_TPL_BOOL_SEL(STATIC_UB, 1),
    ),
);
#endif

#pragma pack(push, 8)
struct alignas(8) RecurrentGatedDeltaRuleTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t t;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t b;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
    uint32_t hasGama;
    uint32_t hasGamaK;
    uint32_t hasAcceptedTokens;
    uint32_t stateStride0;
 	uint32_t stateStride1;
 	uint32_t stateStride2;
};
#pragma pack(pop)

// Byte offsets in the A5 case 1 single-head tile. Each short gate/beta
// buffer has a full register of space even though only 32 elements are valid.
struct RGDRStaticUbLayout {
    static constexpr uint32_t WIDTH = 128;
    static constexpr uint32_t KEY_HEADS = 16;
    static constexpr uint32_t HEADS = 32;
    static constexpr uint32_t REG_BYTES = 256;
    static constexpr uint32_t STATE_ELEMENTS = WIDTH * WIDTH;
    static constexpr uint32_t STATE_BF16_BYTES = STATE_ELEMENTS * sizeof(uint16_t);
    static constexpr uint32_t ROW_BF16_BYTES = WIDTH * sizeof(uint16_t);
    static constexpr uint32_t ROW_FP32_BYTES = WIDTH * sizeof(float);
    static constexpr uint32_t STATE_IN = 0;
    static constexpr uint32_t STATE_SCRATCH = STATE_IN + STATE_BF16_BYTES;
    static constexpr uint32_t STATE_OUT = STATE_SCRATCH + STATE_ELEMENTS * sizeof(float);
    static constexpr uint32_t Q_IN = STATE_OUT + STATE_BF16_BYTES;
    static constexpr uint32_t K_IN = Q_IN + ROW_BF16_BYTES;
    static constexpr uint32_t V_IN = K_IN + ROW_BF16_BYTES;
    static constexpr uint32_t Q_FP32 = V_IN + ROW_BF16_BYTES;
    static constexpr uint32_t K_FP32 = Q_FP32 + ROW_FP32_BYTES;
    static constexpr uint32_t V_FP32 = K_FP32 + ROW_FP32_BYTES;
    static constexpr uint32_t ATTN_TMP = V_FP32 + ROW_FP32_BYTES;
    static constexpr uint32_t ATTN_OUT = ATTN_TMP + ROW_FP32_BYTES;
    static constexpr uint32_t BETA_IN = ATTN_OUT + ROW_BF16_BYTES;
    static constexpr uint32_t GATE_IN = BETA_IN + REG_BYTES;
    static constexpr uint32_t BETA_FP32 = GATE_IN + REG_BYTES;
    static constexpr uint32_t GATE_FP32 = BETA_FP32 + REG_BYTES;
    static constexpr uint32_t BYTES = GATE_FP32 + REG_BYTES;
};

inline constexpr bool CanUseA5StaticUb(const RecurrentGatedDeltaRuleTilingData &t, bool isA5, bool isBf16State)
{
    using L = RGDRStaticUbLayout;
    return isA5 && isBf16State && t.b == 1 && t.t == 1 &&
           t.nk == L::KEY_HEADS && t.nv == L::HEADS && t.vectorCoreNum == L::HEADS &&
           t.dk == L::WIDTH && t.dv == L::WIDTH && t.vStep == L::WIDTH && t.stateStride2 == L::WIDTH &&
           t.stateStride1 == L::STATE_ELEMENTS && t.stateStride0 == L::HEADS * L::STATE_ELEMENTS &&
           t.hasGama == 1 && t.hasGamaK == 0 && t.hasAcceptedTokens == 0 && t.ubCalSize >= L::BYTES;
}

} // namespace RecurrentGatedDeltaRule

#endif // RECURRENT_GATED_DELTA_RULE_STRUCT_H
