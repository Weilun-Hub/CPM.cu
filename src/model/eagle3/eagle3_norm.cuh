#pragma once

#include "../../trait.cuh"
#include "../../utils.cuh"

#include <cuda_runtime.h>

namespace {

template <typename T, typename T2>
__global__ void rms_norm2_kernel(int dim, const T2* input_hidden, const T2* hidden_weight, const T2* input_embed, const T2* embed_weight, T2* output, float eps) {
    __shared__ float warp_sum[32];

    float R_hidden[4][2];
    float R_embed[4][2];

    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum_hidden = 0.0f;
    float sum_embed = 0.0f;

    int iIter = 0;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val_hidden = input_hidden[row * dim + i];
        float val1_hidden = float(val_hidden.x);
        float val2_hidden = float(val_hidden.y);
        
        R_hidden[iIter][0] = val1_hidden;
        R_hidden[iIter][1] = val2_hidden;

        sum_hidden = __fmaf_rn(val1_hidden, val1_hidden, sum_hidden);
        sum_hidden = __fmaf_rn(val2_hidden, val2_hidden, sum_hidden);

        T2 val_embed = input_embed[row * dim + i];
        float val1_embed = float(val_embed.x);
        float val2_embed = float(val_embed.y);

        R_embed[iIter][0] = val1_embed;
        R_embed[iIter][1] = val2_embed;

        sum_embed = __fmaf_rn(val1_embed, val1_embed, sum_embed);
        sum_embed = __fmaf_rn(val2_embed, val2_embed, sum_embed);

        ++iIter;
    }

    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 16);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 8);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 4);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 2);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 1);

    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 16);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 8);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 4);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 2);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 1);

    if ((col & 31) == 0) { 
        int offset = col >> 5;
        warp_sum[offset] = sum_hidden;
        warp_sum[offset + 16] = sum_embed; 
    }

    __syncthreads();

    if (col < 32) {
        float warp_sum_ = warp_sum[col];
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 8, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 4, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 2, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 1, 16);

        float factor = __frcp_rn(dim << 1);
        if ((col == 0) || (col == 16)) {
            warp_sum[col] = __frsqrt_rn(__fmaf_rn(warp_sum_, factor, eps));
        }
    }
    
    __syncthreads();
    
    sum_hidden = warp_sum[0];
    sum_embed = warp_sum[16];
    
    iIter = 0;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 w_hidden = hidden_weight[i];
        T2 w_embed = embed_weight[i];

        int offset = row * dim * 2 + i;
        output[offset + dim] = T2(
            T(sum_hidden * R_hidden[iIter][0] * float(w_hidden.x)),
            T(sum_hidden * R_hidden[iIter][1] * float(w_hidden.y))
        );
        output[offset] = T2(
            T(sum_embed * R_embed[iIter][0] * float(w_embed.x)),
            T(sum_embed * R_embed[iIter][1] * float(w_embed.y))
        );
        ++iIter;
    }
}

template <typename T, typename T2>
__global__ void add_and_rms_norm2_kernel(int dim, const T2* input_hidden, const T2* prev_hidden, const T2* hidden_weight, const T2* input_embed, const T2* embed_weight, T2* output, float eps) {
    __shared__ float warp_sum[32];

    float R_hidden[4][2];
    float R_embed[4][2];

    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum_hidden = 0.0f;
    float sum_embed = 0.0f;

    int iIter = 0;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val_hidden = input_hidden[row * dim + i];
        T2 prev = prev_hidden[row * dim + i];
        val_hidden = val_hidden + prev;
        
        float val1_hidden = float(val_hidden.x);
        float val2_hidden = float(val_hidden.y);
        
        R_hidden[iIter][0] = val1_hidden;
        R_hidden[iIter][1] = val2_hidden;

        sum_hidden = __fmaf_rn(val1_hidden, val1_hidden, sum_hidden);
        sum_hidden = __fmaf_rn(val2_hidden, val2_hidden, sum_hidden);

        T2 val_embed = input_embed[row * dim + i];
        float val1_embed = float(val_embed.x);
        float val2_embed = float(val_embed.y);

        R_embed[iIter][0] = val1_embed;
        R_embed[iIter][1] = val2_embed;

        sum_embed = __fmaf_rn(val1_embed, val1_embed, sum_embed);
        sum_embed = __fmaf_rn(val2_embed, val2_embed, sum_embed);

        ++iIter;
    }

    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 16);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 8);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 4);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 2);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 1);

    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 16);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 8);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 4);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 2);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 1);

    if ((col & 31) == 0) { 
        int offset = col >> 5;
        warp_sum[offset] = sum_hidden;
        warp_sum[offset + 16] = sum_embed; 
    }

    __syncthreads();

    if (col < 32) {
        float warp_sum_ = warp_sum[col];
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 8, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 4, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 2, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 1, 16);

        float factor = __frcp_rn(dim << 1);
        if ((col == 0) || (col == 16)) {
            warp_sum[col] = __frsqrt_rn(__fmaf_rn(warp_sum_, factor, eps));
        }
    }
    
    __syncthreads();
    
    sum_hidden = warp_sum[0];
    sum_embed = warp_sum[16];
    
    iIter = 0;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 w_hidden = hidden_weight[i];
        T2 w_embed = embed_weight[i];

        int offset = row * dim * 2 + i;
        output[offset + dim] = T2(
            T(sum_hidden * R_hidden[iIter][0] * float(w_hidden.x)),
            T(sum_hidden * R_hidden[iIter][1] * float(w_hidden.y))
        );
        output[offset] = T2(
            T(sum_embed * R_embed[iIter][0] * float(w_embed.x)),
            T(sum_embed * R_embed[iIter][1] * float(w_embed.y))
        );
        ++iIter;
    }
}

template <typename T>
void rms_norm2(const Stream& stream, int num_tokens, int dim, const T* input_hidden, const T* hidden_weight, const T* input_embed, const T* embed_weight, T* output, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    rms_norm2_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>((dim >> 1), (T2*)input_hidden, (T2*)hidden_weight, (T2*)input_embed, (T2*)embed_weight, (T2*)output, eps);
}

template <typename T>
void add_and_rms_norm2(const Stream& stream, int num_tokens, int dim, const T* input_hidden, const T* prev_hidden, const T* hidden_weight, const T* input_embed, const T* embed_weight, T* output, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    add_and_rms_norm2_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>((dim >> 1), (T2*)input_hidden, (T2*)prev_hidden, (T2*)hidden_weight, (T2*)input_embed, (T2*)embed_weight, (T2*)output, eps);
}
}

template <typename T>
struct Eagle3Norm {
    int dim;
    float eps;
    T* hidden_weight;
    T* embed_weight;

    T* output;

    Eagle3Norm(int dim, float eps) {
        this->dim = dim;
        this->eps = eps;
    }

    void init_weight_ptr(Memory* memory) {
        hidden_weight = (T*)memory->allocate_for_model(dim * sizeof(T));
        embed_weight = (T*)memory->allocate_for_model(dim * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * 2 * dim * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("hidden_norm") != std::string::npos) {
            cudaMemcpy((void*)hidden_weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("input_layernorm") != std::string::npos) {
            cudaMemcpy((void*)embed_weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input_hidden, T* input_embed, T* prev_hidden, T* tgt=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        if (prev_hidden == nullptr) {
            rms_norm2(stream, num_tokens, this->dim, input_hidden, this->hidden_weight, input_embed, this->embed_weight, tgt, this->eps);
        } else {
            add_and_rms_norm2(stream, num_tokens, this->dim, input_hidden, prev_hidden, this->hidden_weight, input_embed, this->embed_weight, tgt, this->eps);
        }
    }
};