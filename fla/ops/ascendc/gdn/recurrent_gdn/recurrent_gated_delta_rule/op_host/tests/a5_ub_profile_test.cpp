// Standalone host-only contract test; no CANN or NPU is required.
// c++ -std=c++17 -O2 -Wall -Wextra -Werror a5_ub_profile_test.cpp -o a5_ub_profile_test
#define TORCH_MODE
#include <cassert>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>
#include "../recurrent_gated_delta_rule_a5_ub.h"

using namespace RecurrentGatedDeltaRule;

// Independent allocation ledger: append the actual kernel objects in order,
// rather than using the selector's aggregate coefficients.
static std::pair<uint64_t, uint64_t> AllocationLedger(const RecurrentGatedDeltaRuleTilingData &t,
                                                     uint32_t stateBytes, const A5UbProfile &p)
{
    std::vector<uint64_t> queues{2048, 2048, 2048, 128ULL * p.vStep * stateBytes};
    if (t.hasGama != 0) {
        queues.push_back(8ULL * t.nv * 4);
    }
    if (t.hasGamaK != 0) {
        queues.push_back(8 * 128 * 4);
    }
    queues.push_back(8ULL * t.nv * 2);
    for (uint32_t slot = 0; slot < p.stateOutBufferNum; ++slot) {
        queues.push_back(128ULL * p.vStep * stateBytes);
    }
    for (uint32_t slot = 0; slot < p.attnOutBufferNum; ++slot) {
        queues.push_back(2ULL * p.vStep);
    }
    uint64_t queueEnd = 0;
    for (uint64_t size : queues) {
        assert(queueEnd % 32 == 0);
        queueEnd += (size + 31) / 32 * 32;
    }
    const uint64_t gateElements = (8ULL * t.nv + 15) / 16 * 16;
    const std::vector<uint64_t> scratch{
        4ULL * p.vStep, 4096, 4096, 4096, 128ULL * p.vStep * 4,
        gateElements * 4, gateElements * 4};
    uint64_t tmpEnd = 0;
    for (uint64_t size : scratch) {
        assert(tmpEnd % 32 == 0);
        tmpEnd += size;
    }
    return {queueEnd, tmpEnd};
}

static void CheckSelection(const RecurrentGatedDeltaRuleTilingData &t, uint32_t stateBytes, uint64_t ubBytes)
{
    const auto p = SelectA5UbProfile(t, stateBytes, ubBytes);
    assert(p.valid);
    const auto ledger = AllocationLedger(t, stateBytes, p);
    assert(p.queueBytes == ledger.first && p.tmpBytes == ledger.second);
    assert(p.totalBytes == ledger.first + ledger.second + 128);
    assert(p.totalBytes <= ubBytes && p.tmpBytes % 32 == 0);
    assert(p.vStep >= 8 && p.vStep <= 128 && p.vStep % 8 == 0);
    assert(p.repeatTime == (128 + p.vStep - 1) / p.vStep);
    assert(p.vStep == ((128 + p.repeatTime - 1) / p.repeatTime + 7) / 8 * 8);
    if (t.t == 1) {
        assert(p.stateOutBufferNum == 1 && p.attnOutBufferNum == 1);
    }
    // No allowed profile with fewer slices may fit in the same UB.
    for (uint32_t ds = 1; ds <= 2; ++ds) {
        for (uint32_t da = ds; da <= 2; ++da) {
            if (t.t == 1 && (ds != 1 || da != 1)) {
                continue;
            }
            for (uint32_t rows = 8; rows <= 128; rows += 8) {
                A5UbProfile candidate;
                candidate.vStep = rows;
                candidate.stateOutBufferNum = ds;
                candidate.attnOutBufferNum = da;
                const auto bytes = AllocationLedger(t, stateBytes, candidate);
                if (bytes.first + bytes.second + 128 <= ubBytes) {
                    const uint32_t repeats = (128 + rows - 1) / rows;
                    assert(p.repeatTime <= repeats);
                    if (p.repeatTime == repeats) {
                        assert(p.stateOutBufferNum + p.attnOutBufferNum >= ds + da);
                    }
                }
            }
        }
    }
}

int main()
{
    RecurrentGatedDeltaRuleTilingData t{};
    t.dk = t.dv = 128;
    for (uint32_t nv = 1; nv <= 256; ++nv) {
        t.nv = nv;
        for (uint32_t stateBytes : {2U, 4U}) {
            for (uint32_t gates = 0; gates < 4; ++gates) {
                t.hasGama = gates & 1U;
                t.hasGamaK = (gates >> 1) & 1U;
                for (uint32_t tokens : {1U, 2U, 8U, 16U}) {
                    t.t = tokens;
                    for (uint64_t ubBytes : {192ULL * 1024, 248ULL * 1024, 256ULL * 1024}) {
                        CheckSelection(t, stateBytes, ubBytes);
                    }
                    A5UbProfile minimum;
                    minimum.vStep = 8;
                    const auto bytes = AllocationLedger(t, stateBytes, minimum);
                    const uint64_t threshold = bytes.first + bytes.second + 128;
                    assert(!SelectA5UbProfile(t, stateBytes, threshold - 1).valid);
                    CheckSelection(t, stateBytes, threshold);
                }
            }
        }
    }
    t = {};
    t.t = 1;
    t.dk = t.dv = 128;
    t.nv = 32;
    t.hasGama = 1;
    auto p = SelectA5UbProfile(t, 4, 256 * 1024);
    assert(p.valid && p.vStep == 128 && p.stateOutBufferNum == 1 && p.attnOutBufferNum == 1);
    assert(p.totalBytes == 219520 && p.tmpBytes == 80384);
    // T == B is not a single-token contract: [prefix=0, L0=0, L1=2] is legal.
    t.t = t.b = 2;
    p = SelectA5UbProfile(t, 4, 512 * 1024);
    assert(p.valid && p.vStep == 128 && p.stateOutBufferNum == 2 && p.attnOutBufferNum == 2);
    t.dk = 64;
    assert(!SelectA5UbProfile(t, 4, 512 * 1024).valid);
    t.dk = 128;
    assert(!SelectA5UbProfile(t, 1, 512 * 1024).valid);
    std::cout << "A5 UB allocation, profile, T-bound and capacity-boundary contracts passed\n";
}
