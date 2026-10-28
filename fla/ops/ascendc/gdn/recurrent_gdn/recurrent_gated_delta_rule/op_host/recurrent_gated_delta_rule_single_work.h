/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef RECURRENT_GATED_DELTA_RULE_SINGLE_WORK_H
#define RECURRENT_GATED_DELTA_RULE_SINGLE_WORK_H

#include "../op_kernel/recurrent_gated_delta_rule_struct.h"

namespace RecurrentGatedDeltaRule {

// Shapes and flags only. Sequence values and the state index remain device inputs.
inline bool CanUseA5SingleWork(const RecurrentGatedDeltaRuleTilingData &t, bool isA5, bool isBf16State)
{
    return isA5 && isBf16State && t.vectorCoreNum == 1 &&
           t.b == 1 && t.t == 1 && t.nk == 1 && t.nv == 1 &&
           t.dk == 128 && t.dv == 128 && t.vStep == 128 && t.stateStride2 == 128 &&
           t.hasGama == 1 && t.hasGamaK == 0 && t.hasAcceptedTokens == 0;
}

} // namespace RecurrentGatedDeltaRule
#endif
