import importlib.util
import itertools
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('gen_recurrent_gated_delta_rule.py')
module_spec = importlib.util.spec_from_file_location('rgdr_case_generator', source)
generator = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(generator)
base_specs = generator.build_specs()
v2_count = sum(str(s['design_id']).startswith('RGDR-V') for s in base_specs)
profiles = []
for i, (state_dtype, g_dtype, beta_dtype) in enumerate(itertools.product(('bf16', 'fp32'), repeat=3)):
    options = dict(HK=16, HV=48, state_dtype=state_dtype, g_dtype=g_dtype,
                   beta_dtype=beta_dtype, seq_lengths=[[1], [5], [8]][i % 3],
                   state_layout=('head_block_padded' if i % 2 else 'contiguous'),
                   input_layout=('noncontiguous_all' if i % 2 else 'contiguous'),
                   data_profile='traceable_gva')
    if i in (1, 5):
        options.update(use_gate_in_kernel=True, with_dt_bias=(i == 1))
    if i in (2, 6):
        options.update(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True)
    if i in (3, 7):
        options.update(use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
    if i == 4:
        options.update(ssm_state_indices_rank=2, state_index_width=5,
                       state_indices=[[4, 0, 3, 1, 2]], accepted_tokens=[3])
    p = generator._profile(f'target_dtype_{state_dtype}_{g_dtype}_{beta_dtype}', **options)
    profiles.append(p)
profiles.append(generator._profile('target_no_gate_prefix', HK=16, HV=48,
    seq_lengths=[5], prefix_tokens=1, gate_mode='none', state_dtype='bf16',
    state_layout='noncontiguous', input_layout='noncontiguous_qkv',
    use_qk_l2norm_in_kernel=True))
profiles.append(generator._profile('target_rank2_accepted_fp32_padded', HK=16, HV=48,
    seq_lengths=[8], state_dtype='fp32', g_dtype='fp32', beta_dtype='fp32',
    state_layout='head_block_padded', input_layout='noncontiguous_all',
    ssm_state_indices_rank=2, state_index_width=8,
    state_indices=[[7, 0, 6, 1, 5, 2, 4, 3]], accepted_tokens=[5]))
profiles.append(generator._profile('target_bf16_noncontiguous_8', HK=16, HV=48,
    seq_lengths=[8], state_dtype='bf16', state_layout='noncontiguous',
    input_layout='noncontiguous_qkv_beta_g', gate_mode='g'))
for i, p in enumerate(profiles):
    p.update(design_id=f'RGDR-V{v2_count + i + 1:03}', tags='accuracy,generalization,v2,positive',
             platform_focus='ascend950', soc='ascend950')
specs = [generator._finalize(p, len(base_specs) + i) for i, p in enumerate(profiles)]
generator._validate_specs(base_specs + specs)
payloads = [generator._case_payload(s["case_id"], s) for s in specs]
out = Path(__file__).with_name('atk_recurrent_gated_delta_rule_vsplit.json')
out.write_text(json.dumps(payloads, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
for s in specs:
    print(s['case_id'], s['name'], 'T=', s['T'], 'state=', s['state_dtype'],
          'g=', s.get('g_dtype','fp32') if s['gate_mode'] == 'g' else 'none',
          'beta=', s.get('beta_dtype','bf16'), 'layout=', s['state_layout'],
          'input=', s['input_layout'], 'flags=',
          [k for k in ('use_gate_in_kernel','use_qk_l2norm_in_kernel',
                       'use_beta_sigmoid_in_kernel','allow_neg_eigval') if s.get(k)],
          'rank=', s.get('ssm_state_indices_rank',1), 'accepted=', s.get('accepted_tokens'),
          'prefix=',s.get('prefix_tokens',0))
print('OUTPUT:', out)
