# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        recurrent_gated_delta_rule_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    set(RGDR_TIMER_SWITCH "1")
    if(DEFINED ENV{FLA_RGDR_ENABLE_TIMER})
        set(RGDR_TIMER_SWITCH "$ENV{FLA_RGDR_ENABLE_TIMER}")
    endif()
    if(NOT RGDR_TIMER_SWITCH MATCHES "^[01]$")
        message(FATAL_ERROR "FLA_RGDR_ENABLE_TIMER must be 0 or 1")
    endif()
    add_ops_compile_options(
        OP_NAME RecurrentGatedDeltaRule
        OPTIONS -DENABLE_TIMER=${RGDR_TIMER_SWITCH}
                --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
    )
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME RecurrentGatedDeltaRule
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    endif()
else()
    add_modules_sources(OPTYPE recurrent_gated_delta_rule ACLNNTYPE aclnn_exclude)
endif()

