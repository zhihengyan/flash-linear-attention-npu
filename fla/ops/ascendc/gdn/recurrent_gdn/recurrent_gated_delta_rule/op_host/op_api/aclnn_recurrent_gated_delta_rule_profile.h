#ifndef ACLNN_RECURRENT_GATED_DELTA_RULE_PROFILE_H
#define ACLNN_RECURRENT_GATED_DELTA_RULE_PROFILE_H

// Experimental diagnostic symbols. Existing ACLNN entry points are unchanged.
// Direct C callers must provide a zeroed, same-device, 128B-aligned int64 timer
// that does not alias any input/output. L2 validates size/dtype/contiguity;
// the private stable adapter additionally enforces alignment/device/no-alias.
#include "aclnn_recurrent_gated_delta_rule.h"

#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API uint64_t aclnnRecurrentGatedDeltaRuleTimerBufferSize(); // int64 elements
ACLNN_API aclnnStatus aclnnRecurrentGatedDeltaRuleProfileGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta, aclTensor *stateRef,
    const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices, const aclTensor *g, const aclTensor *gk,
    const aclTensor *numAcceptedTokens, aclTensor *timer, float scaleValue, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnRecurrentGatedDeltaRuleProfile(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
