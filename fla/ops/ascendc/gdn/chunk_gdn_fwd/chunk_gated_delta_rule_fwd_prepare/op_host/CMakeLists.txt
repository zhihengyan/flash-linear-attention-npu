# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_fwd_prepare_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_fwd_prepare ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwdPrepare
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch22
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35
)

# Diagnostic-only timer build. Normal builds keep the original workspace and
# remove all timer instructions. Example: PREPARE_TIMER_DIAGNOSTIC=1 bash build.sh ...
if("$ENV{PREPARE_TIMER_DIAGNOSTIC}" STREQUAL "1")
    set(PREPARE_TIMER_DEFINITIONS PREPARE_TIMER_DIAGNOSTIC=1 ENABLE_TIMER=1)
else()
    set(PREPARE_TIMER_DEFINITIONS PREPARE_TIMER_DIAGNOSTIC=0 ENABLE_TIMER=0)
endif()
if("$ENV{PREPARE_EARLY_NEGL_FREE}" STREQUAL "1")
    list(APPEND PREPARE_TIMER_DEFINITIONS PREPARE_EARLY_NEGL_FREE=1)
else()
    list(APPEND PREPARE_TIMER_DEFINITIONS PREPARE_EARLY_NEGL_FREE=0)
endif()
add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwdPrepare
    COMPILE_DEFINITIONS ${PREPARE_TIMER_DEFINITIONS}
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkGatedDeltaRuleFwdPrepare
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
