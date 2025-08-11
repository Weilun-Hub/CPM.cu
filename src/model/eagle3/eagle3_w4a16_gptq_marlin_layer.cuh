#pragma once

#include "../../qgemm/gptq_marlin/gptq_marlin.cuh"
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_ffn.cuh"

#include "eagle3_w4a16_gptq_marlin_attn.cuh"
#include "eagle3_layer.cuh"

#include <cuda_runtime.h>

template <typename T>
struct Eagle3W4A16GPTQMarlinLayer {
    Eagle3W4A16GPTQMarlinAttention<T> *attn;
    W4A16GPTQMarlinGatedFFN<T> *ffn;

    T* output;

    int intermediate_size;
    T* a_tmp;
    float* c_tmp;

    int hidden_size;
    float residual_scale;

    Eagle3W4A16GPTQMarlinLayer(int hidden_size, int intermediate_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps, int group_size, float residual_scale = 1.0, int window_size = 0, bool use_qk_norm = false, bool use_attn_bias = false) {
        this->intermediate_size = intermediate_size;
        this->attn = new Eagle3W4A16GPTQMarlinAttention<T>(hidden_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, group_size, window_size, use_qk_norm, use_attn_bias);
        this->ffn = new W4A16GPTQMarlinGatedFFN<T>(hidden_size, intermediate_size, rms_norm_eps, group_size);
        this->hidden_size = hidden_size;
        this->residual_scale = residual_scale;
    }

    void init_weight_ptr(Memory* memory) {
        this->attn->init_weight_ptr(memory);
        this->ffn->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t a_tmp_offset = memory->allocate((void**)&this->a_tmp, offset, 2* num_tokens * intermediate_size * sizeof(T));
        int reduce_max_m = marlin::determine_reduce_max_m(num_tokens, marlin::max_par);
        int reduce_n = 2*intermediate_size;
        int64_t c_tmp_offset = memory->allocate((void**)&this->c_tmp, a_tmp_offset, reduce_max_m * reduce_n * sizeof(float));

        int64_t attn_end = this->attn->init_output_ptr(memory, num_tokens, c_tmp_offset);
        int64_t ffn_end = this->ffn->init_output_ptr(memory, num_tokens, c_tmp_offset);
        this->output = this->ffn->output;
        return std::max(attn_end, ffn_end);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("attn") != std::string::npos || name.find("input_layernorm") != std::string::npos || name.find("hidden_norm") != std::string::npos) {
            this->attn->load_to_storage(name, ptr);
        } else if (name.find("mlp") != std::string::npos || name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, T* hidden, T* embed, T* prev_hidden, int32_t* position_ids, KVCache<T>* kv_cache, T* prev_layer_states=nullptr) {
        if (prev_hidden != nullptr) {
            elementwise_scale(calc_stream, num_tokens, this->hidden_size, prev_hidden, this->residual_scale);
        }
        cuda_perf_start_on_stream_f(Q_PREFILL_ATTN, calc_stream.stream);
        this->attn->prefill(calc_stream, num_tokens, num_history_tokens, hidden, embed, prev_hidden, position_ids, kv_cache, this->a_tmp, this->c_tmp);
        cuda_perf_stop_on_stream_f(Q_PREFILL_ATTN, calc_stream.stream);
        if (prev_layer_states != nullptr) {
            cudaMemcpyAsync(
                prev_layer_states,    // dst
                hidden,                // src
                num_tokens * this->attn->hidden_size * sizeof(T),
                cudaMemcpyDeviceToDevice,
                calc_stream.stream
            );
        }
        elementwise_scale(calc_stream, num_tokens, this->hidden_size, this->attn->output, this->residual_scale);
        
        cuda_perf_start_on_stream_f(Q_PREFILL_FFN, calc_stream.stream);
        this->ffn->prefill(calc_stream, num_tokens, hidden, this->attn->output, this->a_tmp, this->c_tmp);
        cuda_perf_stop_on_stream_f(Q_PREFILL_FFN, calc_stream.stream);
    }

    void decode(int32_t num_tokens, int32_t padded_length, T* hidden, T* embed, T* prev_hidden, int32_t* position_ids, int32_t* cache_length, const Mask& mask, KVCache<T>* kv_cache, T* prev_layer_states=nullptr) {
        if (prev_hidden != nullptr) {
            elementwise_scale(calc_stream, num_tokens, this->hidden_size, prev_hidden, this->residual_scale);
        }
        cuda_perf_start_on_stream_f(Q_DECODE_ATTN, calc_stream.stream);
        this->attn->decode(calc_stream, num_tokens, padded_length, hidden, embed, prev_hidden, position_ids, cache_length, mask, kv_cache, this->a_tmp, this->c_tmp);
        cuda_perf_stop_on_stream_f(Q_DECODE_ATTN, calc_stream.stream);
        if (prev_layer_states != nullptr) {
            cudaMemcpyAsync(
                prev_layer_states,    // dst
                hidden,                // src
                num_tokens * this->attn->hidden_size * sizeof(T),
                cudaMemcpyDeviceToDevice,
                calc_stream.stream
            );
        }
        elementwise_scale(calc_stream, num_tokens, this->hidden_size, this->attn->output, this->residual_scale);
        cuda_perf_start_on_stream_f(Q_DECODE_FFN, calc_stream.stream);
        this->ffn->decode(calc_stream, num_tokens, hidden, this->attn->output, this->a_tmp, this->c_tmp);
        cuda_perf_stop_on_stream_f(Q_DECODE_FFN, calc_stream.stream);
    }
};
