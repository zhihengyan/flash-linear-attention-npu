"""Offline source contracts; optional C++ control checks are not NPU validation."""
from pathlib import Path
import re
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


def restore_state_index_read(process):
    process = re.sub(
        r"const uint64_t stateOutIndex = seq1 - seq0 == 1\s*"
        r"\? stateOffset : static_cast<uint64_t>\(ssmStateIndicesGm_\.GetValue\(seq_i\)\);", "", process)
    return process.replace("* stateOutIndex +", "* ssmStateIndicesGm_.GetValue(seq_i) +")


class MultiheadSourceTest(unittest.TestCase):
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
        process = restore_state_index_read(body(source, "ProcessHead"))
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
                     "PrefetchState", "LoadPrefetchedState", "CopyOutState", "CopyOutAttn", "IsCurrentBlock"):
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

    def test_length_reuse_preserves_load_math_and_all_sequence_guards(self):
        source, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL)
        avg = body(source, "ComputeAvgload")
        self.assertIn("__aicore__ inline int32_t ComputeAvgload()", source)
        avg = avg.replace("int32_t lastSeqLen = 0;", "").replace("return lastSeqLen;", "")
        avg = re.sub(r"lastSeqLen = cuSeqlensGm_\.GetValue\(batch_i\);\s*realT \+= lastSeqLen;",
                     "realT += cuSeqlensGm_.GetValue(batch_i);", avg)
        self.assertEqual(normalize(avg), normalize(body(old, "ComputeAvgload")))
        process = body(source, "ProcessGeneral")
        process = process.replace("const int32_t lastSeqLen = ComputeAvgload();", "ComputeAvgload();")
        process = process.replace("B_ == 1 ? lastSeqLen : ", "")
        self.assertEqual(normalize(process), normalize(body(old, "ProcessGeneral")))

    def test_state_index_reuse_is_restricted_to_one_token(self):
        source = KERNEL.read_text(encoding="utf-8")
        original = subprocess.check_output(["git", "show", "a6d33dbb:"+KERNEL.relative_to(ROOT).as_posix()],
                                           cwd=ROOT, text=True, encoding="utf-8")
        self.assertEqual(normalize(restore_state_index_read(body(source, "ProcessHead"))),
                         normalize(body(original, "ProcessHead")))


class MetadataControlTest(unittest.TestCase):
    def test_production_control_matches_baseline_offsets_and_reduces_reads(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("C++17 compiler unavailable; production scalar harness was not executed")
        source = KERNEL.read_text(encoding="utf-8")
        old = subprocess.check_output(["git", "show", "a6d33dbb:"+KERNEL.relative_to(ROOT).as_posix()],
                                      cwd=ROOT, text=True, encoding="utf-8")
        support = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
struct Tensor {
    std::vector<int32_t> values;
    std::vector<uint64_t> reads;
    int32_t GetValue(uint64_t i) { reads.push_back(i); return values.at(i); }
};
struct Local { float GetValue(uint64_t i) { return float(i + 1); } };
struct Queue { void FreeTensor(Local&) {} };
struct Control {
    using inType = int;
    uint32_t B_=0,T_=0,NV_=32,NK_=16,alignK_=128,realK_=128,alignV_=128,realV_=128,vStep_=128;
    uint32_t load=0,usedblk=0,avgload=0,cores=32,blockIdx=0;
    uint32_t stateStride0_=1<<28,stateStride1_=20000,stateStride2_=144;
    uint32_t attnOutBufferNum_=1,stateOutBufferNum_=1;
    static constexpr uint32_t BUFFER_NUM=1,MAX_MTP=8;
    bool hasAcceptedTokens_=false,hasGama_=true,hasGamaK_=false;
    float gama_=0,beta_=0;
    Tensor cuSeqlensGm_,ssmStateIndicesGm_,numAcceptedTokensGm_;
    Local gamaInUb,betaInUb,gamaKInUb;
    Queue gamaKInQueue_;
    std::vector<std::vector<uint64_t>> events;
    uint32_t GetBlockNum() {return cores;}
    uint64_t Ceil(uint64_t a,uint64_t b) {return (a+b-1)/b;}
    void CopyInGamaBeta(int32_t a,int32_t b) {events.push_back({1,uint64_t(a),uint64_t(b)});}
    void CopyInQKV(uint64_t a,uint64_t b,int32_t c) {events.push_back({2,a,b,uint64_t(c)});}
    void PrefetchState(uint64_t a,uint32_t b) {events.push_back({3,a,b});}
    void LoadPrefetchedState(uint32_t a) {events.push_back({4,a});}
    template<bool store> void Compute(uint32_t a,uint64_t b,uint64_t c) {
        events.push_back({5,a,b,c,store,uint64_t(gama_),uint64_t(beta_)});
    }
    void CopyOutAttn(uint64_t a,uint32_t b) {events.push_back({6,a,b});}
    void CopyOutState(uint64_t a,uint32_t b) {events.push_back({7,a,b});}
};
'''
        def control(name, text, return_type):
            signatures = ((return_type, "ComputeAvgload", ""), ("bool", "IsCurrentBlock", "int32_t seqlen"),
                          ("void", "ProcessHead", "int32_t seq0,int32_t seq1,uint64_t head_i,uint64_t stateOffset"),
                          ("void", "ProcessGeneral", ""))
            return "struct "+name+" : Control {\n"+"\n".join(
                result+" "+function+"("+arguments+") {"+body(text, function)+"}"
                for result, function, arguments in signatures)+"\n};\n"
        main = r'''
int main() {
    size_t checked=0;
    const std::vector<std::vector<int32_t>> sequences={
        {},{0},{1},{2},{8},{-1},{9},{0,1},{1,0},{0,1,0},{1,2,1},{8,1,0},{0,0,0},{1,1,1}};
    for(const auto& seq:sequences) for(int prefix:{0,3,-1}) for(int acceptedMode:{0,1,2,3,4})
    for(uint32_t cores:{1u,5u,32u}) for(uint32_t nv:{2u,32u}) for(uint32_t step:{64u,128u})
    for(uint32_t depth:{1u,2u}) for(uint32_t core=0;core<cores;++core) {
        Control config;
        config.B_=seq.size();config.T_=std::max(prefix,0);
        config.cuSeqlensGm_.values.push_back(prefix);
        for(auto length:seq) {
            config.cuSeqlensGm_.values.push_back(length);config.T_+=std::max(length,0);
            config.numAcceptedTokensGm_.values.push_back(acceptedMode==1 ? 1 :
                acceptedMode==2 ? std::max(1,length) : acceptedMode==3 ? 0 : length+1);
        }
        config.ssmStateIndicesGm_.values.resize(config.T_);
        for(uint32_t t=0;t<config.T_;++t)config.ssmStateIndicesGm_.values[t]=config.T_-t+7;
        config.hasAcceptedTokens_=acceptedMode!=0;
        config.NV_=nv;config.NK_=nv/2;config.cores=cores;config.blockIdx=core;config.vStep_=step;
        config.attnOutBufferNum_=depth;config.stateOutBufferNum_=depth;
        config.hasGama_=checked%2;config.hasGamaK_=checked%3==0;
        Baseline before;Candidate after;
        static_cast<Control&>(before)=config;static_cast<Control&>(after)=config;
        before.ProcessGeneral();after.ProcessGeneral();
        if(before.events!=after.events)throw std::runtime_error("address/order/gate trace differs");
        assert(before.avgload==after.avgload && before.load==after.load && before.usedblk==after.usedblk);
        assert(after.cuSeqlensGm_.reads.size()<=before.cuSeqlensGm_.reads.size());
        assert(after.ssmStateIndicesGm_.reads.size()<=before.ssmStateIndicesGm_.reads.size());
        assert(before.numAcceptedTokensGm_.reads==after.numAcceptedTokensGm_.reads);
        if(seq==std::vector<int32_t>{1} && prefix==0 && cores==32 && nv==32 && step==128 && acceptedMode==0) {
            assert(before.cuSeqlensGm_.reads.size()==3 && after.cuSeqlensGm_.reads.size()==2);
            assert(before.ssmStateIndicesGm_.reads.size()==2 && after.ssmStateIndicesGm_.reads.size()==1);
        }
        ++checked;
    }
    std::cout<<checked<<" production scalar-control cases passed\n";
}
'''
        code = support+control("Baseline", old, "void")+control("Candidate", source, "int32_t")+main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"metadata_control.cpp"
            executable = Path(directory)/"metadata_control.exe"
            path.write_text(code, encoding="utf-8")
            build = subprocess.run([compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout+build.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("production scalar-control cases passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
