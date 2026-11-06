/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef RECURRENT_GATED_DELTA_RULE_STATIC_UB_H
#define RECURRENT_GATED_DELTA_RULE_STATIC_UB_H

#include "recurrent_gated_delta_rule.h"

namespace RecurrentGatedDeltaRule {

template <typename T>
__aicore__ inline __ubuf__ T *StaticUbAddress(const LocalTensor<T> &tensor)
{
    return reinterpret_cast<__ubuf__ T *>(tensor.GetPhyAddr());
}

__aicore__ inline void ProcessStaticUb(const RGDRInitParams &params, const RecurrentGatedDeltaRuleTilingData &tiling)
{
    using L = RGDRStaticUbLayout;
    static_assert(sizeof(bfloat16_t) == sizeof(uint16_t) && L::REG_BYTES == VECTOR_REG_WIDTH);
    static_assert(L::BYTES == 132 * 1024);
    AscendC::InitSocState();

    const uint32_t head = AscendC::GetBlockIdx();
    GlobalTensor<int32_t> seqLengths;
    seqLengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.cuSeqlens));
    if (head >= L::HEADS || seqLengths.GetValue(0) != 0 || seqLengths.GetValue(1) != 1) {
        return;
    }
    GlobalTensor<int32_t> stateIndices;
    stateIndices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ssmStateIndices));
    const uint64_t stateIndex = stateIndices.GetValue(0);
    const uint64_t stateOffset = stateIndex * tiling.stateStride0 + static_cast<uint64_t>(head) * tiling.stateStride1;
    const uint32_t qkOffset = head / (L::HEADS / L::KEY_HEADS) * L::WIDTH;
    const uint32_t vOffset = head * L::WIDTH;

    GlobalTensor<bfloat16_t> queryGm, keyGm, valueGm, betaGm, stateInGm, stateOutGm, attnOutGm;
    GlobalTensor<float> gateGm;
    queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.query));
    keyGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.key));
    valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.value));
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.beta));
    stateInGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.initState));
    stateOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.finalState));
    attnOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.attnOut));
    gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.gama));

    LocalTensor<bfloat16_t> betaIn(TPosition::VECIN, L::BETA_IN, L::REG_BYTES / sizeof(bfloat16_t));
    LocalTensor<float> gateIn(TPosition::VECIN, L::GATE_IN, L::REG_BYTES / sizeof(float));
    LocalTensor<float> betaFp32(TPosition::VECCALC, L::BETA_FP32, L::REG_BYTES / sizeof(float));
    LocalTensor<float> gateFp32(TPosition::VECCALC, L::GATE_FP32, L::REG_BYTES / sizeof(float));
    DataCopyPadParams scalarPad{false, 0, 0, 0};
    DataCopyPad(betaIn, betaGm, DataCopyParams{1, L::HEADS * sizeof(bfloat16_t), 0, 0}, scalarPad);
    DataCopyPad(gateIn, gateGm, DataCopyParams{1, L::HEADS * sizeof(float), 0, 0}, scalarPad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(betaFp32, betaIn, AscendC::RoundMode::CAST_NONE, L::HEADS);
    UnaryRepeatParams expParams{1, 1, FP32_NUM_PER_BLOCK, FP32_NUM_PER_BLOCK};
    Exp(gateFp32, gateIn, static_cast<uint64_t>(L::HEADS), 1, expParams);
    // Publish the gate before waiting on state input so Scalar can read it independently.
    SetFlag<HardEvent::V_S>(EVENT_ID3);

    LocalTensor<bfloat16_t> qIn(TPosition::VECIN, L::Q_IN, L::WIDTH);
    LocalTensor<bfloat16_t> kIn(TPosition::VECIN, L::K_IN, L::WIDTH);
    LocalTensor<bfloat16_t> vIn(TPosition::VECIN, L::V_IN, L::WIDTH);
    LocalTensor<float> qFp32(TPosition::VECCALC, L::Q_FP32, L::WIDTH);
    LocalTensor<float> kFp32(TPosition::VECCALC, L::K_FP32, L::WIDTH);
    LocalTensor<float> vFp32(TPosition::VECCALC, L::V_FP32, L::WIDTH);
    DataCopyExtParams rowParams{1, L::ROW_BF16_BYTES, 0, 0, 0};
    DataCopyPadExtParams<bfloat16_t> rowPad{true, 0, 0, 0};
    DataCopyPad(qIn, queryGm[qkOffset], rowParams, rowPad);
    DataCopyPad(kIn, keyGm[qkOffset], rowParams, rowPad);
    DataCopyPad(vIn, valueGm[vOffset], rowParams, rowPad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Cast(qFp32, qIn, AscendC::RoundMode::CAST_NONE, L::WIDTH);
    Cast(kFp32, kIn, AscendC::RoundMode::CAST_NONE, L::WIDTH);
    Cast(vFp32, vIn, AscendC::RoundMode::CAST_NONE, L::WIDTH);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(qFp32, qFp32, tiling.scale, L::WIDTH);

    LocalTensor<bfloat16_t> stateIn(TPosition::VECIN, L::STATE_IN, L::STATE_ELEMENTS);
    DataCopyExtParams stateInParams{L::WIDTH, L::ROW_BF16_BYTES, 0, 0, 0};
    DataCopyPad(stateIn, stateInGm[stateOffset], stateInParams, rowPad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID3);
    const float gama = gateFp32.GetValue(head);
    const float beta = betaFp32.GetValue(head);

    LocalTensor<float> stateScratch(TPosition::VECCALC, L::STATE_SCRATCH, L::STATE_ELEMENTS);
    LocalTensor<float> attnTmp(TPosition::VECCALC, L::ATTN_TMP, L::WIDTH);
    LocalTensor<bfloat16_t> stateOut(TPosition::VECOUT, L::STATE_OUT, L::STATE_ELEMENTS);
    LocalTensor<bfloat16_t> attnOut(TPosition::VECOUT, L::ATTN_OUT, L::WIDTH);
    AscendC::PipeBarrier<PIPE_V>();
    ComputeRecurrentGatedDeltaRuleVF<bfloat16_t, bfloat16_t, true, false, false, true>(
        StaticUbAddress(stateScratch), StaticUbAddress(kFp32), StaticUbAddress(qFp32), StaticUbAddress(vFp32),
        StaticUbAddress(stateScratch), StaticUbAddress(stateOut), StaticUbAddress(attnOut), StaticUbAddress(attnTmp),
        L::WIDTH, L::WIDTH, gama, beta, StaticUbAddress(stateIn));

    SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
    DataCopyPad(attnOutGm[vOffset], attnOut, DataCopyParams{1, L::ROW_BF16_BYTES, 0, 0});
    DataCopyPad(stateOutGm[stateOffset], stateOut, DataCopyParams{L::WIDTH, L::ROW_BF16_BYTES, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace RecurrentGatedDeltaRule

#endif
