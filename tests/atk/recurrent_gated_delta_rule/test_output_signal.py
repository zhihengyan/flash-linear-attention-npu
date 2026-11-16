"""Supplement ATK for G088/G089 with an out-only signal-sensitivity check.

The existing ATK comparison remains mandatory. This additional check requires
RMS error below 10% of the golden output's RMS (error energy below 1% of signal
energy); all-zero and sign-flipped outputs are negative controls.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

MAX_NORMALIZED_RMS_ERROR = 0.10


def normalized_rms_error(actual, expected):
    actual = actual.detach().cpu().to(torch.float32)
    expected = expected.detach().cpu().to(torch.float32)
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise AssertionError('Invalid output shape or non-finite output')
    reference_rms = expected.square().mean().sqrt()
    if not torch.isfinite(reference_rms) or reference_rms.item() == 0:
        raise AssertionError('Golden output must have finite, nonzero RMS')
    return ((actual - expected).square().mean().sqrt() / reference_rms).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(args.suite_dir.resolve()))
    from executor_recurrent_gated_delta_rule import run_cpu, run_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    marker = SimpleNamespace(kwargs={'marker': torch.zeros(1, device=f'npu:{args.device}')})
    cases = json.loads((args.suite_dir / 'atk_recurrent_gated_delta_rule.json').read_text())
    for case_id in (169, 187, 188):
        case = next(case for case in cases if case['id'] == case_id)
        spec = json.loads(next(item['range_values'] for item in case['inputs'] if item['name'] == 'case_spec'))
        expected_id = {169: 'RGDR-G070', 187: 'RGDR-G088', 188: 'RGDR-G089'}[case_id]
        if spec.get('design_id') != expected_id:
            raise AssertionError(f'case {case_id}: unexpected design ID {spec.get("design_id")}')
        golden, _ = run_cpu(spec)
        output, _ = run_npu(spec, marker)
        torch.npu.synchronize()
        errors = {
            'npu': normalized_rms_error(output, golden),
            'zero': normalized_rms_error(torch.zeros_like(golden), golden),
            'sign_flip': normalized_rms_error(-golden, golden),
        }
        print(f'id={case_id} NRMSE={errors} limit={MAX_NORMALIZED_RMS_ERROR}', flush=True)
        if errors['npu'] >= MAX_NORMALIZED_RMS_ERROR:
            raise AssertionError(f'id {case_id}: NPU output fails signal check')
        if any(errors[name] < MAX_NORMALIZED_RMS_ERROR for name in ('zero', 'sign_flip')):
            raise AssertionError(f'id {case_id}: a negative control was not detected')
    print('PASS: NPU signal checks and negative controls')


if __name__ == '__main__':
    main()
