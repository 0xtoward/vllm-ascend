/*
 * Copyright (c) 2026, vLLM-Ascend contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "kernel_operator.h"
#include "adv_api/index/arithprogression.h"
#include "adv_api/normalization/layernorm.h"

namespace {

constexpr uint32_t kMaxChannels = 1024;
constexpr uint32_t kChannelTransposeTile = 16;
constexpr uint32_t kTimeTile = 8;
constexpr uint32_t kMaxTileElements = kMaxChannels * kTimeTile;
constexpr uint32_t kGatherTileElements = kChannelTransposeTile * kTimeTile;
constexpr uint32_t kLayerNormTmpBytes = 8256;

class ChannelLayerNormMishKernel {
public:
    __aicore__ inline void Init(__gm__ float* x, __gm__ float* weight,
                               __gm__ float* bias, __gm__ float* y,
                               uint32_t batch, uint32_t channels,
                               uint32_t time, uint32_t output_time,
                               float epsilon)
    {
        x_.SetGlobalBuffer(x);
        weight_.SetGlobalBuffer(weight);
        bias_.SetGlobalBuffer(bias);
        y_.SetGlobalBuffer(y);
        batch_ = batch;
        channels_ = channels;
        time_ = time;
        output_time_ = output_time;
        epsilon_ = epsilon;
        tiles_per_batch_ = (time + kTimeTile - 1) / kTimeTile;

        pipe_.InitBuffer(ncl_buf_, kMaxTileElements * sizeof(float));
        pipe_.InitBuffer(btc_buf_, kMaxTileElements * sizeof(float));
        pipe_.InitBuffer(tmp_buf_, kMaxTileElements * sizeof(float));
        pipe_.InitBuffer(weight_buf_, kMaxChannels * sizeof(float));
        pipe_.InitBuffer(bias_buf_, kMaxChannels * sizeof(float));
        pipe_.InitBuffer(reduce_buf_, kMaxChannels * sizeof(float));
        pipe_.InitBuffer(gather_index_buf_,
                         kGatherTileElements * sizeof(uint32_t));
        pipe_.InitBuffer(layernorm_buf_, kLayerNormTmpBytes);

        auto weight_local = weight_buf_.Get<float>();
        auto bias_local = bias_buf_.Get<float>();
        AscendC::DataCopy(weight_local, weight_, channels_);
        AscendC::DataCopy(bias_local, bias_, channels_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);

        auto gather_index = gather_index_buf_.Get<int32_t>();
        for (uint32_t channel = 0; channel < kChannelTransposeTile;
             ++channel) {
            AscendC::ArithProgression(
                gather_index[channel * kTimeTile],
                static_cast<int32_t>(channel * sizeof(float)),
                static_cast<int32_t>(channels_ * sizeof(float)),
                kTimeTile);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t work_items = batch_ * tiles_per_batch_;
        for (uint32_t item = AscendC::GetBlockIdx(); item < work_items;
             item += AscendC::GetBlockNum()) {
            ProcessTile(item);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t item)
    {
        const uint32_t batch_idx = item / tiles_per_batch_;
        const uint32_t tile_idx = item % tiles_per_batch_;
        const uint32_t time_start = tile_idx * kTimeTile;
        const uint32_t valid_time =
            time_start + kTimeTile <= time_ ? kTimeTile : time_ - time_start;

        auto ncl = ncl_buf_.Get<float>();
        auto btc = btc_buf_.Get<float>();
        auto tmp = tmp_buf_.Get<float>();

        const uint64_t input_batch_base =
            static_cast<uint64_t>(batch_idx) * channels_ * time_;
        const uint64_t output_batch_base =
            static_cast<uint64_t>(batch_idx) * channels_ * output_time_;
        const uint32_t load_start =
            valid_time == kTimeTile ? time_start : time_ - kTimeTile;
        AscendC::DataCopyExtParams copy_in{
            static_cast<uint16_t>(channels_),
            static_cast<uint32_t>(kTimeTile * sizeof(float)),
            static_cast<uint32_t>((time_ - kTimeTile) * sizeof(float)), 0, 0};
        AscendC::DataCopyPadExtParams<float> copy_pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(ncl, x_[input_batch_base + load_start], copy_in,
                            copy_pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        const uint32_t row_offset = kTimeTile - valid_time;
        auto transpose_dst = valid_time == kTimeTile ? btc : tmp;
        NclToBtc(transpose_dst, ncl);
        if (valid_time != kTimeTile) {
            AscendC::Duplicate(btc, 0.0f, channels_ * kTimeTile);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t row_idx = 0; row_idx < valid_time; ++row_idx) {
                AscendC::Adds(btc[row_idx * channels_],
                              tmp[(row_offset + row_idx) * channels_], 0.0f,
                              channels_);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        auto weight_local = weight_buf_.Get<float>();
        auto bias_local = bias_buf_.Get<float>();
        auto reduce = reduce_buf_.Get<float>();
        auto layernorm_tmp = layernorm_buf_.Get<uint8_t>();

        AscendC::tiling::LayerNormTiling tiling{};
        tiling.bLength = 1;
        tiling.sLength = kTimeTile;
        tiling.hLength = channels_;
        tiling.originalHLength = channels_;
        tiling.inputXSize = channels_ * kTimeTile;
        tiling.meanVarSize = kTimeTile;
        tiling.numberOfTmpBuf = 2;
        tiling.meanTmpTensorPos = 0;
        tiling.meanTmpTensorSize = kTimeTile;
        tiling.varianceTmpTensorPos = kTimeTile;
        tiling.varianceTmpTensorSize = kTimeTile;
        tiling.tmpBufSize = kLayerNormTmpBytes / sizeof(float);
        tiling.oneTmpSize = channels_;
        tiling.firstTmpStartPos = 0;
        tiling.secondTmpStartPos = channels_;
        tiling.thirdTmpStartPos = 2 * channels_;
        tiling.loopRound = kTimeTile;
        tiling.inputRoundSize = channels_;
        tiling.inputTailSize = 0;
        tiling.inputTailPos = channels_ * kTimeTile;
        tiling.meanVarRoundSize = 1;
        tiling.meanVarTailSize = 0;
        tiling.meanVarTailPos = kTimeTile;
        tiling.bshCurLength = channels_;
        tiling.bsCurLength = 1;
        tiling.lastDimValueBack =
            channels_ == 512 ? 0.001953125f : 0.0009765625f;

        AscendC::LayerNorm<float, true>(
            tmp, reduce, reduce[kTimeTile], btc, weight_local, bias_local,
            layernorm_tmp, epsilon_, tiling);
        AscendC::PipeBarrier<PIPE_V>();

        const uint32_t valid_elements = valid_time * channels_;
        AscendC::Exp(btc, tmp, valid_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(btc, btc, 1.0f, valid_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Ln(btc, btc, valid_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Tanh(btc, btc, valid_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp, tmp, btc, valid_elements);
        AscendC::PipeBarrier<PIPE_V>();

        auto transpose_src = tmp;
        if (valid_time != kTimeTile) {
            AscendC::Duplicate(btc, 0.0f, channels_ * kTimeTile);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t row_idx = 0; row_idx < valid_time; ++row_idx) {
                AscendC::Adds(btc[row_idx * channels_],
                              tmp[row_idx * channels_], 0.0f, channels_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            transpose_src = btc;
        }
        BtcToNcl(ncl, transpose_src);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams copy_out{
            static_cast<uint16_t>(channels_),
            static_cast<uint32_t>(kTimeTile * sizeof(float)), 0,
            static_cast<uint32_t>((output_time_ - kTimeTile) * sizeof(float)),
            0};
        AscendC::DataCopyPad(y_[output_batch_base + time_start], ncl,
                            copy_out);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void NclToBtc(AscendC::LocalTensor<float> dst,
                                    AscendC::LocalTensor<float> src)
    {
        AscendC::TransDataTo5HDParams params(false, false, 1, 0, 0);
        for (uint32_t channel_start = 0; channel_start < channels_;
             channel_start += kChannelTransposeTile) {
            uint64_t dst_list[16];
            uint64_t src_list[16];
            for (uint32_t row = 0; row < kTimeTile; ++row) {
                dst_list[2 * row] = static_cast<uint64_t>(
                    dst[row * channels_ + channel_start].GetPhyAddr());
                dst_list[2 * row + 1] = static_cast<uint64_t>(
                    dst[row * channels_ + channel_start + 8].GetPhyAddr());
            }
            for (uint32_t channel = 0; channel < kChannelTransposeTile;
                 ++channel) {
                src_list[channel] = static_cast<uint64_t>(
                    src[(channel_start + channel) * kTimeTile].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<float>(dst_list, src_list, params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void BtcToNcl(AscendC::LocalTensor<float> dst,
                                    AscendC::LocalTensor<float> src)
    {
        auto gather_index = gather_index_buf_.Get<uint32_t>();
        for (uint32_t channel_start = 0; channel_start < channels_;
             channel_start += kChannelTransposeTile) {
            AscendC::Gather(dst[channel_start * kTimeTile], src,
                            gather_index,
                            channel_start * sizeof(float),
                            kGatherTileElements);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<float> x_;
    AscendC::GlobalTensor<float> weight_;
    AscendC::GlobalTensor<float> bias_;
    AscendC::GlobalTensor<float> y_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ncl_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> btc_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weight_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bias_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gather_index_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> layernorm_buf_;
    uint32_t batch_;
    uint32_t channels_;
    uint32_t time_;
    uint32_t output_time_;
    uint32_t tiles_per_batch_;
    float epsilon_;
};

} // namespace

extern "C" __global__ __aicore__ void channel_layer_norm_mish_kernel(
    __gm__ float* x, __gm__ float* weight, __gm__ float* bias,
    __gm__ float* y, uint32_t batch, uint32_t channels, uint32_t time,
    uint32_t output_time, float epsilon)
{
    ChannelLayerNormMishKernel op;
    op.Init(x, weight, bias, y, batch, channels, time, output_time, epsilon);
    op.Process();
}

namespace vllm_ascend {

void channel_layer_norm_mish_impl(void* stream, void* x, void* weight,
                                  void* bias, void* y, uint32_t batch,
                                  uint32_t channels, uint32_t time,
                                  uint32_t output_time, float epsilon,
                                  uint32_t aiv_num)
{
    channel_layer_norm_mish_kernel<<<aiv_num, nullptr, stream>>>(
        static_cast<float*>(x), static_cast<float*>(weight),
        static_cast<float*>(bias), static_cast<float*>(y), batch, channels,
        time, output_time, epsilon);
}

} // namespace vllm_ascend
