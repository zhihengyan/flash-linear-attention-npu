# ChunkGatedDeltaRuleFwdPrepare API

## 公开入口与语义

Python 入口为 `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd_prepare`，经两段式 ACLNN
`aclnnChunkGatedDeltaRuleFwdPrepareGetWorkspaceSize` /
`aclnnChunkGatedDeltaRuleFwdPrepare` 执行。每个 `(batch, value-head, chunk)` 完成可选
Q/K L2Norm、可选 fused gate、chunk-local gate cumsum、可选 beta sigmoid、
`A=(I+L)^{-1}` 以及 `W=A@kbg`、`U=A@vb`。

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepareGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize, bool allowNegEigval, bool useExp2, bool outputA,
    aclTensor *gOut, aclTensor *wOut, aclTensor *uOut, aclTensor *aOut,
    aclTensor *qHatOptional, aclTensor *kHatOptional,
    aclTensor *qRstdOptional, aclTensor *kRstdOptional,
    aclTensor *betaEffOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepare(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

## 参数契约

| 参数 | dtype | Shape / 约束 |
| --- | --- | --- |
| `q/k` | BF16 | 完全同形 `[B,HK,T,K]`，ND |
| `v` | BF16 | `[B,HV,T,V]`，ND |
| `g/beta` | FP32 或 BF16 | `[B,HV,T]`，ND |
| `aLogOptional/dtBiasOptional` | FP32 或 BF16 | `[HV]`；fused gate 参数 |
| `cuSeqlensOptional/chunkIndicesOptional` | INT64 | 必须成对；varlen 容器 `B=1` |
| `gOut` | FP32 | `[B,HV,T]` |
| `wOut/uOut` | 分别同 k/v | `[B,HV,T,K]` / `[B,HV,T,V]` |
| `aOut` | 同 k | `[B,HV,T,chunkSize]`；`outputA=false` 时仍传占位 |
| `qHatOptional/kHatOptional` | 同 q/k | 四个 norm 输出必须同时提供或同时省略 |
| `qRstdOptional/kRstdOptional` | FP32 | `[B,HK,T]` |
| `betaEffOptional` | FP32 | `[B,HV,T]` |

固定 `K=128`、`V∈{128,256}`、`chunkSize=64`，`HV%HK==0` 且
`HV/HK∈{1,2,3,4}`。目标 SoC 为 Ascend 950（arch35）。`useExp2` 只改变 cumsum 后的指数域；
`allowNegEigval=true` 要求提供 `betaEffOptional`，beta 缩放为 `2*sigmoid(beta)`。
输入由 host 连续化，直接输出要求连续 ND。

本轮性能优化不修改上述接口、参数顺序、数学语义、支持范围、CPU 标杆或精度阈值。
