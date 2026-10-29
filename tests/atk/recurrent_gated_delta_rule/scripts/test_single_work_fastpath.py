"""Source contracts and optional C++ tests; no simulated NPU numerical claims."""
import os
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
OP = ROOT / "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule"
BASE = "c9e2f1db"
SCRATCH_BASE = "c36e18ce"
FUSION_BASE = "fec1e18b"
KERNEL = OP / "op_kernel/arch35/recurrent_gated_delta_rule.h"


def body(source, name):
    match = re.search(r"\b" + name + r"\s*\([^)]*\)\s*\{", source)
    if match is None:
        raise AssertionError("missing function: " + name)
    begin, depth = match.end(), 1
    for end in range(begin, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[begin:end]
    raise AssertionError("unclosed function: " + name)


def baseline(path, ref=BASE):
    return subprocess.check_output(["git", "show", ref + ":" + path.relative_to(ROOT).as_posix()],
                                   cwd=ROOT, text=True, encoding="utf-8")


def normalize_cpp(source):
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"\s+", "", source)


def resolve_scratch_guard(source, keep):
    """Expand only the experiment's constexpr guard for source comparison."""
    marker = "if constexpr (writeUpdatedScratch)"
    if source.count(marker) != 1:
        raise AssertionError("expected exactly one scratch-write guard")
    start = source.index(marker)
    begin = source.index("{", start) + 1
    depth = 1
    for end in range(begin, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            inner = source[begin:end] if keep else ""
            return source[:start] + inner + source[end + 1:]
    raise AssertionError("unclosed scratch-write guard")


def resolve_constexpr(source, condition, keep):
    """Expand an experiment's braced constexpr if/else for source comparison."""
    def closing_brace(begin):
        depth = 1
        for end in range(begin + 1, len(source)):
            depth += (source[end] == "{") - (source[end] == "}")
            if depth == 0:
                return end
        raise AssertionError("unclosed constexpr branch")

    marker = "if constexpr (" + condition + ")"
    while marker in source:
        start = source.index(marker)
        begin = source.index("{", start)
        end = closing_brace(begin)
        selected = source[begin + 1:end] if keep else ""
        after = end + 1
        alternate = re.match(r"\s*else\s*\{", source[after:])
        if alternate:
            else_begin = after + alternate.end() - 1
            else_end = closing_brace(else_begin)
            if not keep:
                selected = source[else_begin + 1:else_end]
            after = else_end + 1
        source = source[:start] + selected + source[after:]
    return source


def cpp_contract_source():
    # Execute the production device control body against recording operations.
    # This checks address/control semantics, not VF math or AscendC synchronization.
    process = body(KERNEL.read_text(encoding="utf-8"), "ProcessSingleWork")
    return r'''
#define TORCH_MODE
#define __ubuf__
#include "recurrent_gated_delta_rule_single_work.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>
using namespace RecurrentGatedDeltaRule;
static_assert(sizeof(RecurrentGatedDeltaRuleTilingData) == 80);
static_assert(alignof(RecurrentGatedDeltaRuleTilingData) == 8);
struct Input {
    std::vector<int32_t> values;
    unsigned reads = 0;
    int32_t GetValue(unsigned i) { ++reads; return values.at(i); }
};
// Recording shims check issue order only; they do not model hardware completion.
enum class HardEvent { V_S };
using TEventID = unsigned;
struct Scalar {
    const bool *vectorReady;
    float GetValue(unsigned i) { assert(i == 0 && *vectorReady); return .25f; }
};
struct Pipe {
    std::vector<int> *calls;
    TEventID FetchEventID(HardEvent event) {
        assert(event == HardEvent::V_S); calls->push_back(8); return 3;
    }
};
using stateType = uint16_t;
template <typename T> struct LocalTensor {
    T *data;
    const bool *held;
    T *GetPhyAddr() const { assert(*held); return data; }
};
struct StateQueue {
    std::vector<int> *calls;
    bool held = false;
    stateType storage[128 * 128]{};
    template <typename T> LocalTensor<T> DeQue() {
        static_assert(std::is_same<T, stateType>::value);
        assert(!held); held = true; calls->push_back(4); return {storage, &held};
    }
    template <typename T> void FreeTensor(LocalTensor<T> tensor) {
        assert(held && tensor.data == storage); held = false; calls->push_back(11);
    }
};
struct Harness {
    Input cuSeqlensGm_, ssmStateIndicesGm_;
    std::vector<int> calls;
    bool vectorReady = false, vectorPublished = false;
    Scalar gamaInUb{&vectorReady}, betaInUb{&vectorReady};
    Pipe pipe{&calls};
    StateQueue stateInQueue_{&calls};
    uint64_t stateStride0_;
    float gama_ = 0, beta_ = 0;
    uint64_t inputOffset = 0, outputOffset = 0;
    Pipe *GetTPipePtr() { return &pipe; }
    template <HardEvent event> void SetFlag(TEventID id) {
        static_assert(event == HardEvent::V_S);
        assert(id == 3 && !vectorPublished); vectorPublished = true; calls.push_back(9);
    }
    template <HardEvent event> void WaitFlag(TEventID id) {
        static_assert(event == HardEvent::V_S);
        assert(id == 3 && vectorPublished); vectorPublished = false;
        vectorReady = true; calls.push_back(10);
    }
    void CopyInGamaBeta(int a, int b) { assert(a == 0 && b == 1); calls.push_back(1); }
    void CopyInQKV(int a, int b, int c) { assert(a == 0 && b == 0 && c == 1); calls.push_back(2); }
    void PrefetchState(uint64_t offset, int v) { assert(v == 128); inputOffset = offset; calls.push_back(3); }
    void Compute(int v, int q, int k, stateType *initialState) {
        assert(v == 128 && q == 0 && k == 0 && vectorReady);
        assert(stateInQueue_.held && initialState == stateInQueue_.storage); calls.push_back(5);
    }
    void CopyOutAttn(int off, int v) {
        assert(off == 0 && v == 128 && !stateInQueue_.held); calls.push_back(6);
    }
    void CopyOutState(uint64_t offset, int v) {
        assert(v == 128 && !stateInQueue_.held); outputOffset = offset; calls.push_back(7);
    }
    void ProcessSingleWork() {
''' + process + r'''
    }
};
int main() {
    RecurrentGatedDeltaRuleTilingData t{};
    t.vectorCoreNum = t.b = t.t = t.nk = t.nv = t.hasGama = 1;
    t.dk = t.dv = t.vStep = t.stateStride2 = 128;
    t.stateStride0 = t.stateStride1 = 16384;
    t.stateOutBufferNum = t.attnOutBufferNum = 2;
    assert(CanUseA5SingleWork(t, true, true));
    assert(!CanUseA5SingleWork(t, false, true));
    assert(!CanUseA5SingleWork(t, true, false));
    using Field = uint32_t RecurrentGatedDeltaRuleTilingData::*;
    const Field required[] = {&RecurrentGatedDeltaRuleTilingData::vectorCoreNum,
        &RecurrentGatedDeltaRuleTilingData::b, &RecurrentGatedDeltaRuleTilingData::t,
        &RecurrentGatedDeltaRuleTilingData::nk, &RecurrentGatedDeltaRuleTilingData::nv,
        &RecurrentGatedDeltaRuleTilingData::dk, &RecurrentGatedDeltaRuleTilingData::dv,
        &RecurrentGatedDeltaRuleTilingData::vStep, &RecurrentGatedDeltaRuleTilingData::stateStride2,
        &RecurrentGatedDeltaRuleTilingData::hasGama};
    for (Field field : required) {
        auto bad = t; bad.*field = 0;
        assert(!CanUseA5SingleWork(bad, true, true));
        bad = t; bad.*field += 1;
        assert(!CanUseA5SingleWork(bad, true, true));
    }
    for (Field field : {&RecurrentGatedDeltaRuleTilingData::hasGamaK,
                        &RecurrentGatedDeltaRuleTilingData::hasAcceptedTokens}) {
        auto bad = t; bad.*field = 1;
        assert(!CanUseA5SingleWork(bad, true, true));
    }
    for (unsigned depth : {1u, 2u}) {
        t.stateOutBufferNum = t.attnOutBufferNum = depth;
        assert(CanUseA5SingleWork(t, true, true));
    }
    for (int prefix : {-1, 0, 1, 2}) for (int length : {-1, 0, 1, 2, 8, 9}) {
        for (int index : {0, 1, 7}) for (uint64_t stride : {16384ULL, 20000ULL}) {
            Harness h; h.cuSeqlensGm_.values = {prefix, length};
            h.ssmStateIndicesGm_.values = {index}; h.stateStride0_ = stride;
            h.ProcessSingleWork();
            // Original general-path validity after fixing B=T=1 and no accepted input.
            bool active = length > 0 && length <= 8 && prefix >= 0 && prefix <= 1 && prefix + length <= 1;
            assert(h.cuSeqlensGm_.reads == 2);
            if (active) {
                assert((h.calls == std::vector<int>{1,2,3,4,8,9,10,5,11,6,7}));
                assert(!h.stateInQueue_.held);
                assert(h.vectorReady && !h.vectorPublished);
                assert(h.ssmStateIndicesGm_.reads == 1);
                assert(h.inputOffset == stride * static_cast<uint64_t>(index));
                assert(h.outputOffset == h.inputOffset);
                assert(h.gama_ == .25f && h.beta_ == .25f);
            } else {
                assert(h.calls.empty() && h.ssmStateIndicesGm_.reads == 0);
                assert(!h.vectorReady && !h.vectorPublished && !h.stateInQueue_.held);
            }
        }
    }
    std::cout << "single-work selection/control contracts: PASS\n";
}
'''


class SingleWorkContractTest(unittest.TestCase):
    def test_original_accuracy_and_performance_inputs_and_key_representatives(self):
        directory = ROOT / "tests/atk/recurrent_gated_delta_rule"
        for name in ("atk_recurrent_gated_delta_rule.json", "atk_recurrent_gated_delta_rule_perf.json",
                     "gen_recurrent_gated_delta_rule.py", "reference_recurrent_gated_delta_rule.py",
                     "executor_recurrent_gated_delta_rule.py"):
            path = directory / name
            with self.subTest(path=name):
                self.assertEqual(path.read_text(encoding="utf-8"), baseline(path))
        path = directory / "atk_recurrent_gated_delta_rule_mss.json"
        cases, old = json.loads(path.read_text(encoding="utf-8")), json.loads(baseline(path))
        accuracy = json.loads((directory / "atk_recurrent_gated_delta_rule.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[0], old[0])
        self.assertEqual(cases[1:], accuracy[1:3])

    def test_original_buffers_and_pipeline_helpers_unchanged(self):
        current, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL)
        for name in ("Init", "SetGlobalTensors", "InitLocalBuffers",
                     "CopyInQKV", "PrefetchState", "LoadPrefetchedState", "ExpMasked",
                     "CopyOutAttn", "CopyOutState", "CopyInGamaBeta", "ProcessHead", "IsCurrentBlock"):
            with self.subTest(name=name):
                self.assertEqual(body(current, name), body(old, name))
        self.assertEqual(body(current, "ProcessGeneral"), body(old, "Process"))

    def test_only_second_pass_scratch_stores_are_optional(self):
        current, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL, SCRATCH_BASE)
        self.assertIn("bool writeUpdatedScratch = true,", current)
        vf = resolve_constexpr(body(current, "ComputeRecurrentGatedDeltaRuleVF"), "inputStateBf16", False)
        parent_vf = body(old, "ComputeRecurrentGatedDeltaRuleVF")
        # The full-write version must retain every original operation.
        self.assertEqual(normalize_cpp(resolve_scratch_guard(vf, True)), normalize_cpp(parent_vf))
        update = "Add(state1, state1, update1, fullFp32Mask);"
        output = "RegTensor<float> out0;"
        tail_begin = parent_vf.index(update) + len(update)
        tail_end = parent_vf.index(output, tail_begin)
        expected = parent_vf[:tail_begin] + parent_vf[tail_end:]
        # The no-write version must differ only in that exact tail store block.
        # In particular, first-pass gated scratch and both GM outputs stay intact.
        self.assertEqual(normalize_cpp(resolve_scratch_guard(vf, False)), normalize_cpp(expected))
        self.assertLess(vf.index(update), vf.index("if constexpr (writeUpdatedScratch)"))
        self.assertLess(vf.index("if constexpr (writeUpdatedScratch)"), vf.index(output))

    def test_scratch_policy_is_limited_to_existing_single_work_path(self):
        current, parent = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL, SCRATCH_BASE)
        compute = resolve_constexpr(body(current, "Compute"), "singleWork", False)
        choices = re.findall(r"ComputeRecurrentGatedDeltaRuleVF<stateType, outType, "
                             r"(true|false), (true|false), !singleWork>", compute)
        self.assertCountEqual(choices, [("true", "true"), ("true", "false"),
                                       ("false", "true"), ("false", "false")])
        self.assertEqual(normalize_cpp(compute.replace(", !singleWork>", ">").replace(
            "gama_, beta_, nullptr", "gama_, beta_")), normalize_cpp(body(parent, "Compute")))
        expected_control = body(parent, "ProcessSingleWork").replace(
            "LoadPrefetchedState(128);", "LocalTensor<stateType> stateLocal = stateInQueue_.DeQue<stateType>();")
        expected_control = expected_control.replace("Compute(128, 0, 0);",
            "Compute(128, 0, 0, reinterpret_cast<__ubuf__ stateType *>(stateLocal.GetPhyAddr()));\n"
            "stateInQueue_.FreeTensor(stateLocal);")
        self.assertEqual(normalize_cpp(body(current, "ProcessSingleWork")), normalize_cpp(expected_control))
        for relative in ("op_host/recurrent_gated_delta_rule_single_work.h",
                         "op_host/recurrent_gated_delta_rule_tiling.cpp",
                         "op_kernel/recurrent_gated_delta_rule_struct.h",
                         "op_kernel/recurrent_gated_delta_rule.cpp"):
            path = OP / relative
            with self.subTest(path=relative):
                self.assertEqual(path.read_text(encoding="utf-8"), baseline(path, SCRATCH_BASE))

    def test_tiling_layout_and_public_interfaces_unchanged(self):
        path = OP / "op_kernel/recurrent_gated_delta_rule_struct.h"
        self.assertEqual(path.read_text(encoding="utf-8").split("#pragma pack(push, 8)")[1],
                         baseline(path).split("#pragma pack(push, 8)")[1])
        for relative in ("op_kernel/recurrent_gated_delta_rule.h", "op_host/recurrent_gated_delta_rule_def.cpp",
                         "op_host/recurrent_gated_delta_rule_tiling_processor.h",
                         "op_host/op_api/aclnn_recurrent_gated_delta_rule.h",
                         "op_host/op_api/aclnn_recurrent_gated_delta_rule.cpp"):
            path = OP / relative
            with self.subTest(path=relative):
                self.assertEqual(path.read_text(encoding="utf-8"), baseline(path))

    def test_initial_cast_fusion_preserves_vf_math_and_lane_pairing(self):
        current, parent = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL, FUSION_BASE)
        self.assertIn("bool inputStateBf16 = false>", current)
        for name, layout in (("castTraitB16ToB32", "ZERO"), ("castTraitB16ToB32One", "ONE")):
            self.assertRegex(current, name + r" = \{\s*RegLayout::" + layout +
                             r", SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN\}")
        vf = body(current, "ComputeRecurrentGatedDeltaRuleVF")
        original = body(parent, "ComputeRecurrentGatedDeltaRuleVF")
        self.assertEqual(normalize_cpp(resolve_constexpr(vf, "inputStateBf16", False)), normalize_cpp(original))
        old_load = '''if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(state0, state + rowOffset);
            DataCopy(state1, state + rowOffset + FP32_PER_REG);
        } else {
            DataCopy<float, LoadDist::DIST_DINTLV_B32>(state0, state1, state + rowOffset);
        }'''
        self.assertEqual(original.count(old_load), 2)
        new_load = '''static_assert(std::is_same<stateType, bfloat16_t>::value && hasGama && !hasGamaK &&
                          !writeUpdatedScratch, "direct BF16 input is limited to single work");
            RegTensor<stateType> initialStateReg;
            DataCopy(initialStateReg, initialState + rowOffset);
            Cast<float, stateType, castTraitB16ToB32>(state0, initialStateReg, fullFp32Mask);
            Cast<float, stateType, castTraitB16ToB32One>(state1, initialStateReg, fullFp32Mask);'''
        self.assertEqual(normalize_cpp(resolve_constexpr(vf, "inputStateBf16", True)),
                         normalize_cpp(original.replace(old_load, new_load, 1)))
        special = resolve_constexpr(body(current, "Compute"), "singleWork", True)
        self.assertIn("true, false, !singleWork, true>", special)
        self.assertIn("gama_, beta_, initialStateAddr", special)
        self.assertIn("__ubuf__ stateType *initialStateAddr = nullptr", current)
        self.assertIn("__ubuf__ stateType *initialState)", current)
        self.assertNotIn("GetValue", special)

    def test_finite_bf16_widening_and_even_odd_layout_model(self):
        # Mathematical exactness/layout model only, not an AscendC Cast execution.
        # Cover every finite BF16 encoding, including signed zero and subnormals.
        for bits in range(1 << 16):
            exponent, fraction = (bits >> 7) & 255, bits & 127
            if exponent == 255:
                continue
            magnitude = (math.ldexp(float(fraction), -133) if exponent == 0 else
                         math.ldexp(1.0 + fraction / 128, exponent - 127))
            value = -magnitude if bits & 0x8000 else magnitude
            self.assertEqual(struct.pack('<f', value), struct.pack('<I', bits << 16))
        # RegLayout ZERO/ONE must preserve the original even/odd FP32 deinterleave.
        for row in range(128):
            raw = [((row * 128 + k) * 40503 + 7919) & 65535 for k in range(128)]
            widened = [value << 16 for value in raw]
            even = [raw[2 * lane] << 16 for lane in range(64)]
            odd = [raw[2 * lane + 1] << 16 for lane in range(64)]
            self.assertEqual(even, widened[::2])
            self.assertEqual(odd, widened[1::2])
        self.assertEqual((127 * 128 + 128) * 2, 32768)

    def test_template_selection_and_platform_wiring(self):
        source = (OP / "op_kernel/recurrent_gated_delta_rule_struct.h").read_text(encoding="utf-8")
        choices = re.findall(r"ASCENDC_TPL_DTYPE_SEL\(D_T_STATE,\s*(\w+)\),\s*"
                             r"ASCENDC_TPL_BOOL_SEL\(SINGLE_WORK,\s*([01])\)", source)
        self.assertCountEqual(choices, [("RGDR_TPL_BF16", "0"), ("RGDR_TPL_FP32", "0"), ("RGDR_TPL_BF16", "1")])
        entry = (OP / "op_kernel/recurrent_gated_delta_rule.cpp").read_text(encoding="utf-8")
        self.assertIn("template <int D_T_STATE, bool SINGLE_WORK>", entry)
        self.assertIn("RGDR<bfloat16_t, bfloat16_t, StateType, SINGLE_WORK>", entry)
        host = (OP / "op_host/recurrent_gated_delta_rule_tiling.cpp").read_text(encoding="utf-8")
        self.assertIn("SocVersion::ASCEND950", host)
        self.assertIn("GET_TPL_TILING_KEY(stateDtypeKey, singleWork)", host)

    def test_specialized_control_pairs_sync_before_scalar_reads(self):
        source = body(KERNEL.read_text(encoding="utf-8"), "ProcessSingleWork")
        self.assertNotRegex(source, r"\b(for|while|PipeBarrier)\s*[<(]")
        self.assertEqual(source.count("FetchEventID(HardEvent::V_S)"), 1)
        self.assertEqual(re.findall(r"SetFlag<HardEvent::V_S>\((\w+)\)", source), ["vToS"])
        self.assertEqual(re.findall(r"WaitFlag<HardEvent::V_S>\((\w+)\)", source), ["vToS"])
        markers = (
            "stateInQueue_.DeQue<stateType>()", "FetchEventID(HardEvent::V_S)",
            "SetFlag<HardEvent::V_S>(vToS)", "WaitFlag<HardEvent::V_S>(vToS)",
            "gamaInUb.GetValue(0)", "betaInUb.GetValue(0)", "Compute(128, 0, 0,",
            "stateInQueue_.FreeTensor(stateLocal)", "CopyOutAttn(0", "CopyOutState(stateOffset")
        for marker in markers:
            self.assertIn(marker, source)
        positions = [source.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(source.count("ssmStateIndicesGm_.GetValue"), 1)
        self.assertLess(source.index("if (prefix"), source.index("ssmStateIndicesGm_.GetValue"))
        self.assertLess(source.index("return;"), source.index("FetchEventID"))
        self.assertLess(source.index("return;"), source.index("stateInQueue_.DeQue"))
        self.assertNotIn("LoadPrefetchedState", source)
        self.assertEqual(source.count("stateInQueue_.DeQue"), 1)
        self.assertEqual(source.count("stateInQueue_.FreeTensor"), 1)
        self.assertLess(source.index("stateInQueue_.FreeTensor"), source.index("CopyOutAttn(0"))
        self.assertLess(source.index("Compute(128"), source.index("CopyOutAttn(0"))
        self.assertLess(source.index("CopyOutAttn(0"), source.index("CopyOutState(stateOffset"))

    def test_cpp_selection_and_production_control_body(self):
        command = shlex.split(os.environ["CXX"]) if os.environ.get("CXX") else []
        if not command:
            compiler = next((shutil.which(name) for name in ("c++", "g++", "clang++") if shutil.which(name)), None)
            if compiler is None:
                self.skipTest("no C++ compiler; source checks are not a CANN compilation")
            command = [compiler]
        with tempfile.TemporaryDirectory(prefix=".rgdr-single-work-", dir=ROOT) as temp:
            directory = Path(temp).resolve()
            directory.relative_to(ROOT.resolve())
            source = directory / "contract.cpp"
            source.write_text(cpp_contract_source(), encoding="utf-8")
            binary = directory / ("contract.exe" if os.name == "nt" else "contract")
            subprocess.run(command + ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(OP / "op_host"),
                                      str(source), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
