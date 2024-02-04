#pragma once

#include "Utils.h"
#include "Stdafx.h"
#include "flash_api.h"

EXPORT_API(void) THSFlash_MHA_FWD(const Tensor q, const Tensor k, const Tensor v, const Tensor out_, const Tensor alibi_slopes_, const float p_dropout, const float softmax_scale, const bool is_causal, const int window_size_left, const int window_size_right, const bool return_softmax, const Generator gen_, Tensor* (*allocator)(size_t length));

EXPORT_API(void) THSFlash_MHA_VARLEN_FWD(const Tensor q, const Tensor k, const Tensor v, const Tensor out_, const Tensor cu_seqlens_q, const Tensor cu_seqlens_k, const Tensor seqused_k_, const Tensor alibi_slopes_, const int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors, const bool is_causal, const int window_size_left, const int window_size_right, const bool return_softmax, const Generator gen_, Tensor* (*allocator)(size_t length));

EXPORT_API(void) THSFlash_MHA_BWD(const Tensor dout, const Tensor q, const Tensor k, const Tensor v, const Tensor out, const Tensor softmax_lse, const Tensor dq_, const Tensor dk_, const Tensor dv_, const Tensor alibi_slopes_, const float p_dropout, const float softmax_scale, const bool is_causal, const int window_size_left, const int window_size_right, const bool deterministic, const Generator gen_, const Tensor rng_state_, Tensor* (*allocator)(size_t length));

EXPORT_API(void) THSFlash_MHA_VARLEN_BWD(const Tensor dout, const Tensor q, const Tensor k, const Tensor v, const Tensor out, const Tensor softmax_lse, const Tensor dq_, const Tensor dk_, const Tensor dv_, const Tensor cu_seqlens_q, const Tensor cu_seqlens_k, const Tensor alibi_slopes_, const int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors, const bool is_causal, const int window_size_left, const int window_size_right, const bool deterministic, const Generator gen_, const Tensor rng_state_, Tensor* (*allocator)(size_t length));

EXPORT_API(void) THSFlash_MHA_FWD_KVCACHE(const Tensor q, const Tensor kcache, const Tensor vcache, const Tensor k_, const Tensor v_, const Tensor seqlens_k_, const Tensor rotary_cos_, const Tensor rotary_sin_, const Tensor cache_batch_idx_, const Tensor block_table_, const Tensor alibi_slopes_, const Tensor out_, const float softmax_scale, const bool is_causal, const int window_size_left, const int window_size_right, const bool is_rotary_interleaved, const int num_splits, Tensor* (*allocator)(size_t length));

EXPORT_API(char*) THSFlash_get_and_reset_last_err();