"""Source contracts and optional C++ tests; no simulated NPU numerical claims."""
import os
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
OP = ROOT / "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule"
BASE = "c9e2f1db"
SCRATCH_BASE = "c36e18ce"
GATE_BASE = "fec1e18b"
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
    """Resolve the named experiment conditions, including their optional else."""
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


def gate_vf_body(source, from_ub):
    source = resolve_constexpr(source, "gatesFromUb", from_ub)
    source = re.sub(r"constexpr bool gatesFromUb = [^;]+;", "", source)
    source = re.sub(r"static_assert\(.*?\);", "", source, flags=re.DOTALL)
    if not from_ub:
        source = source.replace("RegTensor<float> gateReg;", "")
    return source


def cpp_contract_source():
    # Execute the production device control body against recording operations.
    # This checks address/control semantics, not VF math or AscendC synchronization.
    process = body(KERNEL.read_text(encoding="utf-8"), "ProcessSingleWork")
    return r'''
#define TORCH_MODE
#include "recurrent_gated_delta_rule_single_work.h"
#include <cassert>
#include <cstdint>
#include <iostream>
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
struct Harness {
    Input cuSeqlensGm_, ssmStateIndicesGm_;
    std::vector<int> calls;
    bool vectorReady = false, vectorPublished = false;
    Scalar gamaInUb{&vectorReady}, betaInUb{&vectorReady};
    Pipe pipe{&calls};
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
    void LoadPrefetchedState(int v) { assert(v == 128); calls.push_back(4); }
    void Compute(int v, int q, int k) { assert(v == 128 && q == 0 && k == 0); calls.push_back(5); }
    void CopyOutAttn(int off, int v) { assert(off == 0 && v == 128); calls.push_back(6); }
    void CopyOutState(uint64_t offset, int v) { assert(v == 128); outputOffset = offset; calls.push_back(7); }
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
                assert((h.calls == std::vector<int>{1,2,3,4,8,9,10,5,6,7}));
                assert(h.vectorReady && !h.vectorPublished);
                assert(h.ssmStateIndicesGm_.reads == 1);
                assert(h.inputOffset == stride * static_cast<uint64_t>(index));
                assert(h.outputOffset == h.inputOffset);
                // The control path no longer reads the UB scalars; VF math is not simulated.
                assert(h.gama_ == 0 && h.beta_ == 0);
            } else {
                assert(h.calls.empty() && h.ssmStateIndicesGm_.reads == 0);
                assert(!h.vectorReady && !h.vectorPublished);
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
        vf = gate_vf_body(body(current, "ComputeRecurrentGatedDeltaRuleVF"), False)
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
        self.assertEqual(normalize_cpp(compute.replace(", !singleWork>", ">")),
                         normalize_cpp(body(parent, "Compute")))
        control = body(parent, "ProcessSingleWork").replace("gama_ = gamaInUb.GetValue(0);", "")
        control = control.replace("beta_ = betaInUb.GetValue(0);", "")
        self.assertEqual(normalize_cpp(body(current, "ProcessSingleWork")), normalize_cpp(control))
        for relative in ("op_host/recurrent_gated_delta_rule_single_work.h",
                         "op_host/recurrent_gated_delta_rule_tiling.cpp",
                         "op_kernel/recurrent_gated_delta_rule_struct.h",
                         "op_kernel/recurrent_gated_delta_rule.cpp"):
            path = OP / relative
            with self.subTest(path=relative):
                self.assertEqual(path.read_text(encoding="utf-8"), baseline(path, SCRATCH_BASE))

    def test_vf_gate_delivery_and_general_fallback(self):
        current, parent = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL, GATE_BASE)
        self.assertIn("typename gateType = float>", current)
        self.assertIn("gateType gama, gateType beta)", current)
        self.assertIn("hasGama && !hasGamaK && !writeUpdatedScratch", current)
        vf = body(current, "ComputeRecurrentGatedDeltaRuleVF")
        parent_vf = body(parent, "ComputeRecurrentGatedDeltaRuleVF")
        self.assertEqual(normalize_cpp(gate_vf_body(vf, False)), normalize_cpp(parent_vf))
        row_loop = "for (uint16_t row = 0; row < rows; ++row)"
        expected = parent_vf.replace(row_loop,
            "RegTensor<float> gateReg;\n"
            "DataCopy<float, LoadDist::DIST_BRC_B32>(gateReg, gama);\n" + row_loop, 1)
        barrier = "LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();"
        expected = expected.replace(barrier, barrier + "\n"
            "DataCopy<float, LoadDist::DIST_BRC_B32>(gateReg, beta);", 1)
        for dst, src, scalar in (("state0", "state0", "gama"), ("state1", "state1", "gama"),
                                 ("delta", "attn", "beta")):
            expected = expected.replace(f"Muls({dst}, {src}, {scalar}, fullFp32Mask);",
                                        f"Mul({dst}, {src}, gateReg, fullFp32Mask);")
        self.assertEqual(normalize_cpp(gate_vf_body(vf, True)), normalize_cpp(expected))
        compute = body(current, "Compute")
        self.assertEqual(normalize_cpp(resolve_constexpr(compute, "singleWork", False)),
                         normalize_cpp(body(parent, "Compute")))
        special = resolve_constexpr(compute, "singleWork", True)
        self.assertIn("gamaAddr = reinterpret_cast<__ubuf__ float *>(gamaInUb.GetPhyAddr())", special)
        self.assertIn("betaAddr = reinterpret_cast<__ubuf__ float *>(betaInUb.GetPhyAddr())", special)
        self.assertIn("true, false, !singleWork, __ubuf__ float *>", special)
        self.assertRegex(special, r"gamaAddr,\s*betaAddr\);")
        self.assertNotIn("gama_, beta_", special)
        self.assertNotIn("GetValue", special)

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

    def test_specialized_control_pairs_sync_before_vf(self):
        source = body(KERNEL.read_text(encoding="utf-8"), "ProcessSingleWork")
        self.assertNotRegex(source, r"\b(for|while|PipeBarrier)\s*[<(]")
        self.assertEqual(source.count("FetchEventID(HardEvent::V_S)"), 1)
        self.assertEqual(re.findall(r"SetFlag<HardEvent::V_S>\((\w+)\)", source), ["vToS"])
        self.assertEqual(re.findall(r"WaitFlag<HardEvent::V_S>\((\w+)\)", source), ["vToS"])
        positions = [source.index(text) for text in (
            "LoadPrefetchedState(128)", "FetchEventID(HardEvent::V_S)",
            "SetFlag<HardEvent::V_S>(vToS)", "WaitFlag<HardEvent::V_S>(vToS)",
            "Compute(128, 0, 0)")]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("gamaInUb.GetValue", source)
        self.assertNotIn("betaInUb.GetValue", source)
        self.assertEqual(source.count("ssmStateIndicesGm_.GetValue"), 1)
        self.assertLess(source.index("if (prefix"), source.index("ssmStateIndicesGm_.GetValue"))
        self.assertLess(source.index("return;"), source.index("FetchEventID"))
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
