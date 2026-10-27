// Private diagnostic adapter. The public recurrent GDN schema is unchanged.
#include "stable/acl_meta.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace {
using torch::stable::Tensor;
constexpr const char* kSchema_profile_recurrent_gated_delta_rule =
    "_profile_recurrent_gated_delta_rule(Tensor query, Tensor key, Tensor value, "
    "Tensor(a!) state, Tensor beta, Tensor actual_seq_lengths, Tensor ssm_state_indices, "
    "Tensor? num_accepted_tokens, Tensor? g, Tensor? gk, Tensor(b!) timer, "
    "float scale, int stream) -> Tensor";

Tensor run_profile_recurrent_gated_delta_rule(
    Tensor query, Tensor key, Tensor value,
    Tensor state, Tensor beta, Tensor actual_seq_lengths,
    Tensor ssm_state_indices, std::optional<Tensor> num_accepted_tokens,
    std::optional<Tensor> g, std::optional<Tensor> gk,
    Tensor timer, double scale, int64_t stream) {
  using namespace fla_npu_stable::stable;
  const auto value_meta = meta_of(value);
  const auto timer_meta = meta_of(timer);
  using TimerSizeFn = uint64_t (*)();
  const auto minimum = reinterpret_cast<TimerSizeFn>(
      fla_npu_stable::Runtime::instance().symbol("aclnnRecurrentGatedDeltaRuleTimerBufferSize"))();
  if (timer_meta.scalar_type != 4 || timer_meta.ndim != 1 || !timer_meta.contiguous ||
      timer_meta.sizes[0] < static_cast<int64_t>(minimum) ||
      timer_meta.device_type != value_meta.device_type || timer_meta.device_index != value_meta.device_index ||
      timer_meta.data == nullptr ||
      (reinterpret_cast<uintptr_t>(timer_meta.data) + timer_meta.storage_offset * 8) % 128 != 0) {
    throw std::runtime_error("RGDR timer must be same-device, 128B-aligned, contiguous int64 with the queried capacity");
  }
  const auto timer_begin = reinterpret_cast<uintptr_t>(timer_meta.data);
  const auto timer_end = timer_begin + timer_meta.storage_numel * 8;
  auto refuse_alias = [&](const TensorMeta& input) {
    if (!input.defined) return;
    const auto begin = reinterpret_cast<uintptr_t>(input.data);
    const auto end = begin + input.storage_numel * element_size(input.scalar_type);
    if (begin < timer_end && timer_begin < end) {
      throw std::runtime_error("RGDR timer storage must not overlap an operator input");
    }
  };
  for (const auto& input : {meta_of(query), meta_of(key), value_meta, meta_of(state), meta_of(beta),
                           meta_of(actual_seq_lengths), meta_of(ssm_state_indices)}) {
    refuse_alias(input);
  }
  if (g) refuse_alias(meta_of(*g));
  if (gk) refuse_alias(meta_of(*gk));
  if (num_accepted_tokens) refuse_alias(meta_of(*num_accepted_tokens));
  auto out = allocate_like(value_meta);
  FLA_STABLE_EXEC("aclnnRecurrentGatedDeltaRuleProfile", value_meta, stream,
                  nd_tensor(meta_of(query)), nd_tensor(meta_of(key)), nd_tensor(value_meta),
                  nd_tensor(meta_of(beta)), nd_out_tensor(meta_of(state)),
                  nd_tensor(meta_of(actual_seq_lengths)), nd_tensor(meta_of(ssm_state_indices)),
                  nd_optional_tensor(g), nd_optional_tensor(gk), nd_optional_tensor(num_accepted_tokens),
                  nd_out_tensor(timer_meta), scalar(static_cast<float>(scale)), nd_out_tensor(meta_of(out)));
  return out;
}
} // namespace
