"""Offline source contracts; optional C++ control checks are not NPU validation."""
from pathlib import Path
import ast
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


RANGE_BLOCK = re.compile(
    r"uint64_t headBegin = 0;\s*uint64_t headEnd = NV_;\s*"
    r"if \(B_ == 1 && seqLen == 1 && NV_ > 1\) \{\s*"
    r"(?://[^\n]*\n\s*)?headBegin = blockIdx \* static_cast<uint64_t>\(avgload\);\s*"
    r"const uint64_t headLimit = headBegin \+ avgload;\s*"
    r"headEnd = headLimit < NV_ \? headLimit : NV_;\s*"
    r"load = 0;\s*usedblk = static_cast<uint32_t>\(blockIdx\);\s*\}")


def production_range_expressions(source):
    process = body(source, "ProcessGeneral")
    active = process.split("if (B_ == 1 && seqLen == 1 && NV_ > 1) {", 1)[1].split("}", 1)[0]
    expressions = []
    for name in ("headBegin", "headLimit", "headEnd"):
        expr = re.search(r"\b"+name+r" = ([^;]+);", active).group(1)
        expr = re.sub(r"static_cast<uint64_t>\((\w+)\)", r"\1", expr)
        if "?" in expr:
            condition, alternatives = expr.split("?", 1)
            yes, no = alternatives.split(":", 1)
            expr = "("+yes+") if ("+condition+") else ("+no+")"
        tree = ast.parse(expr.strip(), mode="eval")
        allowed = (ast.Expression, ast.BinOp, ast.Mult, ast.Add, ast.Compare, ast.Lt,
                   ast.IfExp, ast.Name, ast.Load, ast.Constant)
        if not all(isinstance(node, allowed) for node in ast.walk(tree)):
            raise AssertionError("unexpected non-integer range expression")
        expressions.append((name, compile(tree, "<production head range>", "eval")))
    return expressions


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
                     "PrefetchState", "LoadPrefetchedState", "CopyOutState", "CopyOutAttn", "IsCurrentBlock",
                     "ComputeAvgload", "Process", "RGDR"):
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

    def test_range_only_changes_the_scan_and_preserves_every_sequence_guard(self):
        source, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL)
        process = body(source, "ProcessGeneral")
        restored, count = RANGE_BLOCK.subn("", process)
        self.assertEqual(count, 1)
        restored = restored.replace("head_i = headBegin; head_i < headEnd;", "head_i = 0; head_i < NV_;")
        self.assertEqual(normalize(restored), normalize(body(old, "ProcessGeneral")))
        self.assertGreater(process.index("uint64_t headBegin"), process.index("seq1 += seqLen;"))

    def test_metadata_reuse_is_removed_and_only_general_differs_from_a6(self):
        source = KERNEL.read_text(encoding="utf-8")
        original = subprocess.check_output(["git", "show", "a6d33dbb:"+KERNEL.relative_to(ROOT).as_posix()],
                                           cwd=ROOT, text=True, encoding="utf-8")
        start = source.index("    __aicore__ inline void ProcessGeneral()")
        end = source.index("\nprivate:", start)
        old_start = original.index("    __aicore__ inline void ProcessGeneral()")
        old_end = original.index("\nprivate:", old_start)
        self.assertEqual(source[:start]+source[end:], original[:old_start]+original[old_end:])


class HeadOwnershipTest(unittest.TestCase):
    def test_exhaustive_production_bounds_match_original_cursor_ownership(self):
        expressions = production_range_expressions(KERNEL.read_text(encoding="utf-8"))
        checked = 0
        for heads in range(2, 257):
            for cores in range(1, 97):
                average = (heads+cores-1)//cores
                # Independent reference: the original mutable load/usedblk scan.
                owners = [[] for _ in range(cores)]
                load, owner = 0, 0
                for head in range(heads):
                    load += 1
                    owners[owner].append(head)
                    if load >= average:
                        load, owner = 0, owner+1
                for core in range(cores):
                    values = dict(blockIdx=core, avgload=average, NV_=heads)
                    for name, expression in expressions:
                        values[name] = eval(expression, {"__builtins__": {}}, values)
                    actual = list(range(values["headBegin"], values["headEnd"]))
                    self.assertEqual(actual, owners[core], (heads, cores, core))
                    checked += 1
        self.assertEqual(checked, 1187280)

    def test_case_one_checks_fall_from_1024_to_32_without_new_division(self):
        source = KERNEL.read_text(encoding="utf-8")
        expressions = production_range_expressions(source)
        total = 0
        for core in range(32):
            values = dict(blockIdx=core, avgload=1, NV_=32)
            for name, expression in expressions:
                values[name] = eval(expression, {"__builtins__": {}}, values)
            self.assertEqual(list(range(values["headBegin"], values["headEnd"])), [core])
            total += values["headEnd"]-values["headBegin"]
        self.assertEqual(total, 32)
        active = RANGE_BLOCK.search(body(source, "ProcessGeneral")).group()
        self.assertNotIn("Ceil", active)
        self.assertNotIn("/", re.sub(r"//[^\n]*", "", active))


class HeadRangeControlTest(unittest.TestCase):
    def test_production_control_matches_baseline_events_and_metadata_reads(self):
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
    uint32_t load=0,usedblk=0,avgload=0,cores=32,blockIdx=0,ownershipChecks=0;
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
                result+" "+function+"("+arguments+") {"+
                ("++ownershipChecks;" if function == "IsCurrentBlock" else "")+body(text, function)+"}"
                for result, function, arguments in signatures)+"\n};\n"
        main = r'''
int main() {
    size_t checked=0;
    const std::vector<std::vector<int32_t>> sequences={
        {},{0},{1},{2},{8},{-1},{9},{0,1},{1,0},{0,1,0},{1,2,1},{8,1,0},{0,0,0},{1,1,1}};
    for(const auto& seq:sequences) for(int prefix:{0,3,-1}) for(int acceptedMode:{0,1,2,3,4})
    for(uint32_t cores:{1u,5u,32u}) for(uint32_t nv:{1u,2u,31u,32u}) for(uint32_t step:{64u,128u})
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
        config.NV_=nv;config.NK_=nv%2==0 ? nv/2 : 1;config.cores=cores;config.blockIdx=core;config.vStep_=step;
        config.attnOutBufferNum_=depth;config.stateOutBufferNum_=depth;
        config.hasGama_=checked%2;config.hasGamaK_=checked%3==0;
        Baseline before;Candidate after;
        static_cast<Control&>(before)=config;static_cast<Control&>(after)=config;
        before.ProcessGeneral();after.ProcessGeneral();
        if(before.events!=after.events)throw std::runtime_error("address/order/gate trace differs");
        assert(before.avgload==after.avgload);
        if(seq.size()!=1 || seq[0]!=1 || nv<=1 || prefix<0)
            assert(before.load==after.load && before.usedblk==after.usedblk);
        assert(after.ownershipChecks<=before.ownershipChecks);
        assert(after.cuSeqlensGm_.reads==before.cuSeqlensGm_.reads);
        assert(after.ssmStateIndicesGm_.reads==before.ssmStateIndicesGm_.reads);
        assert(before.numAcceptedTokensGm_.reads==after.numAcceptedTokensGm_.reads);
        if(seq==std::vector<int32_t>{1} && prefix==0 && cores==32 && nv==32 && step==128 && acceptedMode==0) {
            assert(before.ownershipChecks==32 && after.ownershipChecks==1);
        }
        ++checked;
    }
    std::cout<<checked<<" production scalar-control cases passed\n";
}
'''
        code = support+control("Baseline", old, "void")+control("Candidate", source, "void")+main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"head_range_control.cpp"
            executable = Path(directory)/"head_range_control.exe"
            path.write_text(code, encoding="utf-8")
            build = subprocess.run([compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout+build.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("production scalar-control cases passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
