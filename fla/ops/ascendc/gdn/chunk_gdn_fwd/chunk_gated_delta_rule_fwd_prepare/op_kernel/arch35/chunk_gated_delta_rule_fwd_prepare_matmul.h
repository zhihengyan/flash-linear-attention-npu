/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Cube MMAD: C[m, n] = A[m, k] @ B[k, n] into L0C. L1 is cube NZ.
 *   fp32 C0=8, bf16 C0=16.
 *   transposeB=false: B NZ is [n, k] (kkt: same k' tile as A, n=m).
 *   transposeB=true:  B NZ is [k, n] (I / -L / Y). Leaf tiles from 8-col
 *   ND->NZ are NZ(inv.T): pass transposeA / !transposeB for those Loads.
 * fp32 L1 slots are 64x64, so A/B srcStride is kNumMFracs64.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_PREPARE_MATMUL_H
#define CHUNK_GATED_DELTA_RULE_FWD_PREPARE_MATMUL_H

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_fwd_prepare_common.h"

namespace ChunkGatedDeltaRuleFwdPrepare {
using namespace AscendC;

__aicore__ inline void FillLoad2D(AscendC::LoadData2DParamsV2 &p, uint16_t mStep, uint16_t kStep,
                                  uint16_t srcStride, uint16_t dstStride, bool transpose)
{
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = mStep;
    p.kStep = kStep;
    p.srcStride = srcStride;
    p.dstStride = dstStride;
    p.ifTranspose = transpose;
    p.sid = 0;
}

template <typename InDtype>
__aicore__ inline void LoadL1NzToL0AB(AscendC::LocalTensor<InDtype> l1A,
                                      AscendC::LocalTensor<InDtype> l1B,
                                      AscendC::LocalTensor<InDtype> l0A,
                                      AscendC::LocalTensor<InDtype> l0B,
                                      int32_t m, int32_t n, int32_t k,
                                      bool transposeB, bool transposeA, uint8_t evt)
{
    constexpr uint32_t c0 = 32 / sizeof(InDtype);
    constexpr bool kFp32 = sizeof(InDtype) == sizeof(float);
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t aKFracs = static_cast<uint16_t>(k / c0);
    const uint16_t aSrc = kFp32 ? static_cast<uint16_t>(kNumMFracs64) : mFracs;

    AscendC::LoadData2DParamsV2 loadA;
    FillLoad2D(loadA, mFracs, aKFracs, aSrc, mFracs, transposeA);
    AscendC::LoadData(l0A, l1A, loadA);

    const uint16_t bMFracs = transposeB ? static_cast<uint16_t>(k / 16)
                                        : static_cast<uint16_t>(n / 16);
    const uint16_t bKFracs = transposeB ? static_cast<uint16_t>(n / c0)
                                        : static_cast<uint16_t>(k / c0);
    const uint16_t bSrc = kFp32 ? static_cast<uint16_t>(kNumMFracs64) : bMFracs;

    AscendC::LoadData2DParamsV2 loadB;
    FillLoad2D(loadB, bMFracs, bKFracs, bSrc, bMFracs, transposeB);
    AscendC::LoadData(l0B, l1B, loadB);

    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
}

template <typename InDtype>
__aicore__ inline void MmadL0ABToL0C(AscendC::LocalTensor<InDtype> l0A,
                                     AscendC::LocalTensor<InDtype> l0B,
                                     AscendC::LocalTensor<float> l0C,
                                     int32_t m, int32_t n, int32_t k, bool initC, uint8_t evt)
{
    WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    AscendC::MmadParams mmad;
    mmad.m = m;
    mmad.n = n;
    mmad.k = k;
    mmad.cmatrixInitVal = initC;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, mmad);
}

template <typename InDtype>
__aicore__ inline void MatmulToL0C(AscendC::LocalTensor<InDtype> l1A,
                                   AscendC::LocalTensor<InDtype> l1B,
                                   AscendC::LocalTensor<InDtype> l0A,
                                   AscendC::LocalTensor<InDtype> l0B,
                                   AscendC::LocalTensor<float> l0C,
                                   int32_t m, int32_t n, int32_t k,
                                   bool initC, bool transposeB, bool transposeA = false,
                                   uint8_t evt = 0)
{
    LoadL1NzToL0AB<InDtype>(l1A, l1B, l0A, l0B, m, n, k, transposeB, transposeA, evt);
    MmadL0ABToL0C<InDtype>(l0A, l0B, l0C, m, n, k, initC, evt);
}

// C = A[m,k] @ B[k,n]. One MMAD. A is 64x64 (kStep=4). B is 64x128
// (isTranspose, kStep=8, dstStride=8). MMAD k stays kDim.
template <typename InDtype>
__aicore__ inline void WuMatmulToL0C(AscendC::LocalTensor<InDtype> l1A,
                                     AscendC::LocalTensor<InDtype> l1B,
                                     AscendC::LocalTensor<InDtype> l0A,
                                     AscendC::LocalTensor<InDtype> l0B,
                                     AscendC::LocalTensor<float> l0C,
                                     int32_t m, int32_t n, int32_t kDim,
                                     uint8_t evt)
{
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t nFracs = static_cast<uint16_t>(n / 16);
    const uint16_t kFracs = static_cast<uint16_t>(kDim / 16);

    AscendC::LoadData2DParamsV2 loadA;
    FillLoad2D(loadA, mFracs, kFracs, mFracs, mFracs, false);
    AscendC::LoadData(l0A, l1A, loadA);

    AscendC::LoadData2DParamsV2 loadB;
    FillLoad2D(loadB, kFracs, nFracs, kFracs, nFracs, true);
    AscendC::LoadData(l0B, l1B, loadB);

    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    AscendC::MmadParams mmad;
    mmad.m = m;
    mmad.n = n;
    mmad.k = kDim;
    mmad.cmatrixInitVal = true;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, mmad);
}

// [W | U] = A[m,k] @ [kbg | vb]. Both B halves share one packed-NZ L0B
// layout, and the two outputs are extracted from the corresponding L0C columns.
template <typename InDtype>
__aicore__ inline void WuPackedMatmulToUb(AscendC::LocalTensor<InDtype> l1A,
                                          AscendC::LocalTensor<InDtype> l1B0,
                                          AscendC::LocalTensor<InDtype> l1B1,
                                          AscendC::LocalTensor<InDtype> l0A,
                                          AscendC::LocalTensor<InDtype> l0B,
                                          AscendC::LocalTensor<float> l0C,
                                          AscendC::LocalTensor<InDtype> ubW,
                                          AscendC::LocalTensor<InDtype> ubU,
                                          int32_t m, int32_t packedN, int32_t kDim,
                                          int32_t validRows, uint8_t aiv, uint8_t evt)
{
    using ArchTag = Catlass::Arch::Ascend950;
    using Layout = Catlass::layout::RowMajor;
    using TileCopy = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, InDtype, Layout, InDtype, Layout, InDtype, Layout>;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, InDtype, typename TileCopy::LayoutTagL1A>;

    const int32_t halfN = packedN / 2;
    l0B.SetSize(static_cast<uint32_t>(packedN * kDim));
    l0C.SetSize(static_cast<uint32_t>(m * packedN));
    auto tensorL1A = tla::MakeTensor(
        l1A, tla::MakeLayout<InDtype, typename TileCopy::LayoutTagL1A>(m, kDim),
        Catlass::Arch::PositionL1{});
    auto tensorL1B0 = tla::MakeTensor(
        l1B0, tla::MakeLayout<InDtype, typename TileCopy::LayoutTagL1B>(kDim, halfN),
        Catlass::Arch::PositionL1{});
    auto tensorL1B1 = tla::MakeTensor(
        l1B1, tla::MakeLayout<InDtype, typename TileCopy::LayoutTagL1B>(kDim, halfN),
        Catlass::Arch::PositionL1{});
    auto tensorL0A = tla::MakeTensor(
        l0A, tla::MakeLayout<InDtype, typename TileCopy::LayoutTagL0A>(m, kDim),
        Catlass::Arch::PositionL0A{});
    auto tensorL0B = tla::MakeTensor(
        l0B, tla::MakeLayout<InDtype, typename TileCopy::LayoutTagL0B>(kDim, packedN),
        Catlass::Arch::PositionL0B{});
    auto tensorL0C = tla::MakeTensor(
        l0C, tla::MakeLayoutL0C(m, packedN), Catlass::Arch::PositionL0C{});
    auto tensorUbW = tla::MakeTensor(
        ubW, tla::MakeLayout<InDtype, Layout>(m, halfN), Catlass::Arch::PositionUB{});
    auto tensorUbU = tla::MakeTensor(
        ubU, tla::MakeLayout<InDtype, Layout>(m, halfN), Catlass::Arch::PositionUB{});

    auto tileL1A = tla::GetTile(
        tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, kDim));
    auto tileL1B0 = tla::GetTile(
        tensorL1B0, tla::MakeCoord(0, 0), tla::MakeShape(kDim, halfN));
    auto tileL1B1 = tla::GetTile(
        tensorL1B1, tla::MakeCoord(0, 0), tla::MakeShape(kDim, halfN));
    auto tileL0A = tla::GetTile(
        tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, kDim));
    auto tileL0B0 = tla::GetTile(
        tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(kDim, halfN));
    auto tileL0B1 = tla::GetTile(
        tensorL0B, tla::MakeCoord(0, halfN), tla::MakeShape(kDim, halfN));
    auto tileL0B = tla::GetTile(
        tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(kDim, packedN));
    auto tileL0C = tla::GetTile(
        tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, packedN));
    auto tileL0CW = tla::GetTile(
        tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, halfN));
    auto tileL0CU = tla::GetTile(
        tensorL0C, tla::MakeCoord(0, halfN), tla::MakeShape(m, halfN));
    auto blockUbW = tla::GetTile(
        tensorUbW, tla::MakeCoord(0, 0), tla::MakeShape(validRows, halfN));
    auto blockUbU = tla::GetTile(
        tensorUbU, tla::MakeCoord(0, 0), tla::MakeShape(validRows, halfN));

    CopyL1ToL0A copyA;
    CopyL1ToL0B copyB;
    copyA(tileL0A, tileL1A);
    copyB(tileL0B0, tileL1B0);
    copyB(tileL0B1, tileL1B1);
    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    TileMmad mmad;
    // Use the shape-based overload so the bias overload cannot participate.
    mmad(tileL0C, tileL0A, tileL0B, true, evt);
    SetFlag<AscendC::HardEvent::M_MTE1>(evt);
    SetFlag<AscendC::HardEvent::M_FIX>(evt);
    WaitFlag<AscendC::HardEvent::M_FIX>(evt);
    using CopyW = typename TileCopy::template CopyL0CToDst<decltype(blockUbW)>;
    using CopyU = typename TileCopy::template CopyL0CToDst<decltype(blockUbU)>;
    CopyW copyW;
    CopyU copyU;
    copyW(blockUbW, tileL0CW, aiv, 0);
    copyU(blockUbU, tileL0CU, aiv, 0);
    SetFlag<AscendC::HardEvent::FIX_M>(evt);
}

} // namespace ChunkGatedDeltaRuleFwdPrepare

#endif
