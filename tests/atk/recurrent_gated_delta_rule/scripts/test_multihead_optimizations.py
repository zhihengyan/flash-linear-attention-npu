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


def baseline(path, revision=BASE):
    return subprocess.check_output(["git", "show", revision + ":" + path.relative_to(ROOT).as_posix()],
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


def without_bf16_input_branch(source):
    return re.sub(r"^([ ]*)if constexpr \(loadBf16State\) \{\n.*?^\1\} else ",
                  r"\1", source, flags=re.M | re.S)


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
        vf = without_bf16_input_branch(body(source, "ComputeRecurrentGatedDeltaRuleVF"))
        self.assertEqual(normalize(vf), normalize(body(old, "ComputeRecurrentGatedDeltaRuleVF")))
        self.assertEqual(body(source, "ProcessSingleWork"), body(old, "ProcessSingleWork"))

    def test_all_gate_modes_forward_the_compile_time_store_policy(self):
        source, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL)
        self.assertIn("template <bool writeUpdatedScratch = !singleWork, bool loadBf16State = false>", source)
        compute = without_bf16_input_branch(body(source, "Compute"))
        choices = re.findall(r"ComputeRecurrentGatedDeltaRuleVF<stateType, outType, "
                             r"(true|false), (true|false), writeUpdatedScratch>", compute)
        self.assertCountEqual(choices, [("true", "true"), ("true", "false"),
                                       ("false", "true"), ("false", "false")])
        compute = compute.replace(", writeUpdatedScratch>", ", !singleWork>").replace(", beta_, nullptr", ", beta_")
        self.assertEqual(compute, body(old, "Compute"))

    def test_fallback_keeps_original_head_body_and_tail_store_policy(self):
        source, old = KERNEL.read_text(encoding="utf-8"), baseline(KERNEL)
        head = body(source, "ProcessHead")
        before_select = head.split("if constexpr (std::is_same<stateType, bfloat16_t>::value)", 1)[0]
        after_tiles = head.split("ProcessHeadTiles<false>(seq0, seq1, head_i, stateOffset);", 1)[1]
        tiles = without_bf16_input_branch(body(source, "ProcessHeadTiles"))
        tiles = tiles.replace("LocalTensor<stateType> initialState;", "")
        tiles = re.sub(r"\{\s*(LoadPrefetchedState\(curSingleV\);)\s*\}", r"\1", tiles)
        process = before_select + tiles + after_tiles
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

    def test_cast_fusion_keeps_the_adopted_head_scan(self):
        source = KERNEL.read_text(encoding="utf-8")
        self.assertEqual(body(source, "ProcessGeneral"), body(baseline(KERNEL, "0d188aca"), "ProcessGeneral"))

    def test_bf16_input_uses_original_lane_order_and_queue_lifetime(self):
        source = KERNEL.read_text(encoding="utf-8")
        self.assertRegex(source, r"castTraitB16ToB32\s*=\s*\{\s*RegLayout::ZERO,")
        self.assertRegex(source, r"castTraitB16ToB32One\s*=\s*\{\s*RegLayout::ONE,")
        vf = body(source, "ComputeRecurrentGatedDeltaRuleVF")
        load = vf.split("if constexpr (loadBf16State)", 1)[1].split("} else", 1)[0]
        self.assertIn("DataCopy(input, initialState + rowOffset);", load)
        self.assertIn("castTraitB16ToB32>(state0, input, fullFp32Mask);", load)
        self.assertIn("castTraitB16ToB32One>(state1, input, fullFp32Mask);", load)
        tiles = body(source, "ProcessHeadTiles")
        lifetime = ("PrefetchState(nextStateOffset, nextSingleV);", "stateInQueue_.DeQue<stateType>()",
                    "initialState.GetPhyAddr()", "Compute<false, true>", "stateInQueue_.FreeTensor(initialState)")
        positions = [tiles.index(operation) for operation in lifetime]
        self.assertEqual(positions, sorted(positions))


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
#include <type_traits>
#include <vector>
#define __ubuf__
using bfloat16_t = uint16_t;
struct Tensor {
    std::vector<int32_t> values;
    std::vector<uint64_t> reads;
    int32_t GetValue(uint64_t i) { reads.push_back(i); return values.at(i); }
};
struct Local {
    float GetValue(uint64_t i) { return float(i + 1); }
    bfloat16_t* GetPhyAddr() { static bfloat16_t slot; return &slot; }
};
template<typename> using LocalTensor = Local;
struct Queue { void FreeTensor(Local&) {} };
struct StateQueue {
    std::vector<std::vector<uint64_t>>* events = nullptr;
    bool held = false;
    template<typename> Local DeQue() {
        assert(!held);
        held = true;
        events->push_back({8});
        return {};
    }
    void FreeTensor(Local&) {
        assert(held);
        held = false;
        events->push_back({9});
    }
};
struct Control {
    using inType = int;
    using stateType = bfloat16_t;
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
    StateQueue stateInQueue_;
    std::vector<std::vector<uint64_t>> events;
    uint32_t GetBlockNum() {return cores;}
    uint64_t Ceil(uint64_t a,uint64_t b) {return (a+b-1)/b;}
    void CopyInGamaBeta(int32_t a,int32_t b) {events.push_back({1,uint64_t(a),uint64_t(b)});}
    void CopyInQKV(uint64_t a,uint64_t b,int32_t c) {events.push_back({2,a,b,uint64_t(c)});}
    void PrefetchState(uint64_t a,uint32_t b) {events.push_back({3,a,b});}
    void LoadPrefetchedState(uint32_t a) {events.push_back({4,a});}
    template<bool store, bool direct=false>
    void Compute(uint32_t a,uint64_t b,uint64_t c,bfloat16_t* input=nullptr) {
        assert(direct == (input != nullptr));
        assert(direct == stateInQueue_.held);
        assert(!direct || !store);
        events.push_back({5,a,b,c,store,uint64_t(gama_),uint64_t(beta_)});
    }
    void CopyOutAttn(uint64_t a,uint32_t b) {events.push_back({6,a,b});}
    void CopyOutState(uint64_t a,uint32_t b) {events.push_back({7,a,b});}
};
'''
        def control(name, text):
            head_args = "int32_t seq0,int32_t seq1,uint64_t head_i,uint64_t stateOffset"
            signatures = (("void", "ComputeAvgload", ""), ("bool", "IsCurrentBlock", "int32_t seqlen"),
                          ("void", "ProcessHead", head_args),
                          ("void", "ProcessGeneral", ""))
            tiles = ""
            if name == "Candidate":
                tiles = ("template<bool loadBf16State> void ProcessHeadTiles("+head_args+") {"+
                         body(text, "ProcessHeadTiles")+"}\n")
            return "struct "+name+" : Control {\n"+tiles+"\n".join(
                result+" "+function+"("+arguments+") {"+
                ("++ownershipChecks;" if function == "IsCurrentBlock" else "")+body(text, function)+"}"
                for result, function, arguments in signatures)+"\n};\n"
        main = r'''
int main() {
    size_t checked=0;
    size_t reusedDirectSlots=0;
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
        config.attnOutBufferNum_=depth;config.stateOutBufferNum_=checked%3==0 ? 1 : depth;
        config.hasGama_=checked%2;config.hasGamaK_=checked%3==0;
        Baseline before;Candidate after;
        static_cast<Control&>(before)=config;static_cast<Control&>(after)=config;
        after.stateInQueue_.events=&after.events;
        before.ProcessGeneral();after.ProcessGeneral();
        const bool direct=seq==std::vector<int32_t>{1} && prefix==0 && nv>1 && step==128 &&
                          acceptedMode==0 && config.hasGama_ && !config.hasGamaK_;
        std::vector<std::vector<uint64_t>> expected;
        size_t directTiles=0;
        for(const auto& event:before.events) {
            if(direct && event[0]==4) {
                expected.push_back({8});
                ++directTiles;
            } else {
                expected.push_back(event);
            }
            if(direct && event[0]==5)expected.push_back({9});
        }
        if(directTiles>1)++reusedDirectSlots;
        if(expected!=after.events || after.stateInQueue_.held)
            throw std::runtime_error("address/order/gate/slot trace differs");
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
    assert(reusedDirectSlots>0);
    std::cout<<checked<<" production scalar-control cases passed\n";
}
'''
        code = support+control("Baseline", old)+control("Candidate", source)+main
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
