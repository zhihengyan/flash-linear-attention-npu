"""Offline source contracts; optional C++ control checks are not NPU validation."""
from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
OP = ROOT / "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule"
KERNEL = OP / "op_kernel/arch35/recurrent_gated_delta_rule.h"
BASE = "f918240b"


def baseline(path):
    return subprocess.check_output(["git", "show", BASE + ":" + path.relative_to(ROOT).as_posix()],
                                   cwd=ROOT, text=True, encoding="utf-8")


def body(source, name):
    match = re.search(r"\b" + name + r"\s*\([^)]*\)\s*(?:const\s*)?\{", source)
    if match is None:
        raise AssertionError("missing function: " + name)
    begin, depth = match.end(), 1
    for end in range(begin, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[begin:end]
    raise AssertionError("unclosed function: " + name)


def normalize(source):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S))


def control_harness():
    source = KERNEL.read_text(encoding="utf-8")
    return r'''
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>
struct bfloat16_t {};
enum class HardEvent { V_S };
using TEventID = unsigned;
struct Input {
    std::vector<int32_t> values;
    unsigned reads = 0;
    int32_t GetValue(unsigned i) { ++reads; return values.at(i); }
};
template <class stateType = bfloat16_t> struct Harness {
    uint32_t B_=1, T_=1, NK_=16, NV_=32, realK_=128, realV_=128, vStep_=128, stateStride2_=128;
    uint64_t stateStride0_=1048576, stateStride1_=20000, blockIdx=0, cores=32;
    bool hasGama_=true, hasGamaK_=false, hasAcceptedTokens_=false;
    float gama_=0, beta_=0;
    Input cuSeqlensGm_{{0,1}}, ssmStateIndicesGm_{{7}};
    bool published=false, ready=false;
    unsigned fetches=0, waits=0, gateCopies=0;
    uint64_t activeHead=0;
    std::vector<uint64_t> heads, states, queries;
    struct Gate {
        Harness *owner;
        float GetValue(uint64_t h) { assert(owner->ready && h==owner->activeHead); return .25f; }
    } gamaInUb{this}, betaInUb{this};
    uint64_t GetBlockNum() const { return cores; }
    Harness *GetTPipePtr() { return this; }
    TEventID FetchEventID(HardEvent e) { assert(e==HardEvent::V_S); ++fetches; return 3; }
    template <HardEvent e> void SetFlag(TEventID id) {
        static_assert(e==HardEvent::V_S); assert(id==3 && !published); published=true;
    }
    template <HardEvent e> void WaitFlag(TEventID id) {
        static_assert(e==HardEvent::V_S); assert(id==3 && published); published=false; ready=true; ++waits;
    }
    void CopyInGamaBeta(int a, int b) { assert(a==0 && b==1); ++gateCopies; }
    void CopyInQKV(uint64_t v, uint64_t q, int n) {
        assert(n==1 && !published); activeHead=v/128; ready=false; heads.push_back(activeHead); queries.push_back(q);
    }
    void PrefetchState(uint64_t offset, int n) { assert(n==128); states.push_back(offset); }
    void LoadPrefetchedState(int n) { assert(n==128); }
    template <bool keep> void Compute(int n, int q, int v) {
        static_assert(!keep); assert(n==128 && q==0 && v==0 && ready && gama_==.25f && beta_==.25f);
    }
    void CopyOutAttn(uint64_t offset, int n) { assert(n==128 && offset==activeHead*128); }
    void CopyOutState(uint64_t offset, int n) { assert(n==128 && offset==states.back()); }
    bool CanUseSingleTokenHeads() const {
''' + body(source, "CanUseSingleTokenHeads") + r'''
    }
    void ProcessSingleTokenHeads() {
''' + body(source, "ProcessSingleTokenHeads") + r'''
    }
};
int main() {
    assert(Harness<>().CanUseSingleTokenHeads());
    assert(!Harness<float>().CanUseSingleTokenHeads());
    using Field = uint32_t Harness<>::*;
    for (Field f : {&Harness<>::B_, &Harness<>::T_, &Harness<>::realK_, &Harness<>::realV_,
                    &Harness<>::vStep_, &Harness<>::stateStride2_, &Harness<>::NK_, &Harness<>::NV_}) {
        Harness<> h; h.*f=0; assert(!h.CanUseSingleTokenHeads());
        Harness<> other; other.*f+=1; assert(!other.CanUseSingleTokenHeads());
    }
    { Harness<> h; h.hasGama_=false; assert(!h.CanUseSingleTokenHeads()); }
    { Harness<> h; h.hasGamaK_=true; assert(!h.CanUseSingleTokenHeads()); }
    { Harness<> h; h.hasAcceptedTokens_=true; assert(!h.CanUseSingleTokenHeads()); }
    for (unsigned cores : {1u,2u,7u,16u,32u,80u}) for (int index : {0,1,7}) {
        std::vector<unsigned> owners(32,0);
        for (unsigned core=0; core<cores; ++core) {
            Harness<> h; h.cores=cores; h.blockIdx=core; h.ssmStateIndicesGm_.values={index};
            h.ProcessSingleTokenHeads();
            if (core>=32) { assert(h.cuSeqlensGm_.reads==0 && h.fetches==0); continue; }
            assert(h.cuSeqlensGm_.reads==2 && h.ssmStateIndicesGm_.reads==1 && h.fetches==1);
            assert(h.gateCopies==1 && h.waits==h.heads.size() && !h.published);
            for (unsigned i=0; i<h.heads.size(); ++i) {
                auto hv=h.heads[i]; ++owners.at(hv);
                assert(hv%cores==core && h.queries[i]==hv/2*128);
                assert(h.states[i]==h.stateStride0_*index+h.stateStride1_*hv);
            }
        }
        for (auto count : owners) assert(count==1);
    }
    for (int prefix : {-1,0,1,2}) for (int length : {-1,0,1,2,8}) {
        if (prefix==0 && length==1) continue;
        Harness<> h; h.cuSeqlensGm_.values={prefix,length}; h.ProcessSingleTokenHeads();
        assert(h.heads.empty() && h.ssmStateIndicesGm_.reads==0 && h.fetches==0 && h.gateCopies==0);
    }
    { Harness<> h; h.cores=0; h.ProcessSingleTokenHeads(); assert(h.cuSeqlensGm_.reads==0); }
}
'''


class MultiheadSourceTest(unittest.TestCase):
    def test_fast_path_guard_and_general_fallback(self):
        source = KERNEL.read_text(encoding="utf-8")
        expected = '''return std::is_same<stateType, bfloat16_t>::value && B_ == 1 && T_ == 1 &&
            NK_ > 0 && NV_ > 1 && NV_ % NK_ == 0 &&
            realK_ == 128 && realV_ == 128 && vStep_ == 128 && stateStride2_ == 128 &&
            hasGama_ && !hasGamaK_ && !hasAcceptedTokens_;'''
        self.assertEqual(normalize(body(source, "CanUseSingleTokenHeads")), normalize(expected))
        self.assertEqual(normalize(body(source, "Process")), normalize('''
            if constexpr (singleWork) { ProcessSingleWork(); }
            else if (CanUseSingleTokenHeads()) { ProcessSingleTokenHeads(); }
            else { ProcessGeneral(); }'''))

    def test_direct_dispatch_offsets_and_vector_to_scalar_completion(self):
        source = body(KERNEL.read_text(encoding="utf-8"), "ProcessSingleTokenHeads")
        self.assertNotRegex(source, r"\b(?:ComputeAvgload|IsCurrentBlock|ProcessHead)\s*\(")
        self.assertRegex(source, r"for \(uint64_t head_i = blockIdx; head_i < NV_; head_i \+= coreCount\)")
        self.assertIn("(head_i / (NV_ / NK_)) * realK_", source)
        self.assertIn("stateStride0_ * stateIndex + stateStride1_ * head_i", source)
        self.assertEqual(source.count("ssmStateIndicesGm_.GetValue"), 1)
        self.assertEqual(source.count("FetchEventID(HardEvent::V_S)"), 1)
        order = [source.index(text) for text in (
            "if (prefix != 0 || length != 1)", "ssmStateIndicesGm_.GetValue(0)",
            "CopyInGamaBeta(0, 1)", "FetchEventID(HardEvent::V_S)", "for (uint64_t head_i",
            "LoadPrefetchedState(128)", "SetFlag<HardEvent::V_S>(vToS)",
            "WaitFlag<HardEvent::V_S>(vToS)", "gamaInUb.GetValue(head_i)",
            "betaInUb.GetValue(head_i)", "Compute<false>(128, 0, 0)",
            "CopyOutAttn(vOffset, 128)", "CopyOutState(stateOffset, 128)")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(source.count("SetFlag<HardEvent::V_S>"), 1)
        self.assertEqual(source.count("WaitFlag<HardEvent::V_S>"), 1)

    def test_production_control_body_with_optional_cpp_compiler(self):
        command = shlex.split(os.environ["CXX"]) if os.environ.get("CXX") else []
        if not command:
            compiler = next((shutil.which(name) for name in ("c++", "g++", "clang++") if shutil.which(name)), None)
            if compiler is None:
                self.skipTest("no C++ compiler; source checks are not CANN/NPU validation")
            command = [compiler]
        with tempfile.TemporaryDirectory(prefix="rgdr-control-") as temp:
            path = Path(temp) / "control.cpp"
            path.write_text(control_harness(), encoding="utf-8")
            binary = path.with_suffix(".exe" if os.name == "nt" else "")
            subprocess.run(command + ["-std=c++17", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_original_atk_json_files_are_preserved(self):
        for name in ("atk_recurrent_gated_delta_rule.json", "atk_recurrent_gated_delta_rule_perf.json",
                     "atk_recurrent_gated_delta_rule_mss.json"):
            path = ROOT / "tests/atk/recurrent_gated_delta_rule" / name
            with self.subTest(path=name):
                self.assertEqual(path.read_text(encoding="utf-8"), baseline(path))

    def test_vf_math_and_scratch_guard_are_unchanged(self):
        source, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL)
        self.assertEqual(body(source, "ComputeRecurrentGatedDeltaRuleVF"),
                         body(old, "ComputeRecurrentGatedDeltaRuleVF"))
        self.assertEqual(body(source, "ProcessSingleWork"), body(old, "ProcessSingleWork"))

    def test_all_gate_modes_forward_the_compile_time_store_policy(self):
        source, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL)
        self.assertIn("template <bool writeUpdatedScratch = !singleWork>", source)
        compute = body(source, "Compute")
        choices = re.findall(r"ComputeRecurrentGatedDeltaRuleVF<stateType, outType, "
                             r"(true|false), (true|false), writeUpdatedScratch>", compute)
        self.assertCountEqual(choices, [("true", "true"), ("true", "false"),
                                       ("false", "true"), ("false", "false")])
        self.assertEqual(compute.replace(", writeUpdatedScratch>", ", !singleWork>"), body(old, "Compute"))

    def test_general_path_changes_only_the_final_token_compute_dispatch(self):
        source, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL)
        process = body(source, "ProcessHead")
        dispatch = re.compile(
            r"if\s*\(seq_i\s*\+\s*1\s*<\s*static_cast<uint64_t>\(seq1\)\)\s*\{\s*"
            r"Compute<true>\(curSingleV, curQKOffset, curVOffset\);\s*\}\s*else\s*\{\s*"
            r"Compute<false>\(curSingleV, curQKOffset, curVOffset\);\s*\}")
        replaced, count = dispatch.subn("Compute(curSingleV, curQKOffset, curVOffset);", process)
        self.assertEqual(count, 1)
        self.assertEqual(normalize(replaced), normalize(body(old, "ProcessHead")))
        self.assertIn("for (uint64_t v_i", process)
        self.assertIn("for (uint64_t seq_i", process)

    def test_original_storage_helpers_and_public_contracts_are_unchanged(self):
        source, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL)
        for name in ("Init", "SetGlobalTensors", "InitLocalBuffers", "CopyInGamaBeta", "CopyInQKV",
                     "PrefetchState", "LoadPrefetchedState", "CopyOutState", "CopyOutAttn", "ProcessGeneral"):
            with self.subTest(name=name):
                self.assertEqual(body(source, name), body(old, name))
        for relative in ("op_host/op_api/aclnn_recurrent_gated_delta_rule.h",
                         "op_host/op_api/aclnn_recurrent_gated_delta_rule.cpp",
                         "op_host/recurrent_gated_delta_rule_tiling.cpp",
                         "op_host/recurrent_gated_delta_rule_def.cpp",
                         "op_kernel/recurrent_gated_delta_rule_struct.h",
                         "op_kernel/recurrent_gated_delta_rule.cpp",
                         "op_kernel/recurrent_gated_delta_rule.h"):
            path = OP / relative
            with self.subTest(path=relative):
                self.assertEqual(path.read_text(encoding="utf-8"), baseline(path))


if __name__ == "__main__":
    unittest.main()
