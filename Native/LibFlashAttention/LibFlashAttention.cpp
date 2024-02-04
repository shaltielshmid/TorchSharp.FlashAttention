#include "LibFlashAttention.h"

#define OPTIONAL(var, type) var == nullptr ? c10::optional<type>() : *var

void THSFlash_MHA_FWD(const Tensor q, const Tensor k, const Tensor v, const Tensor out_, const Tensor alibi_slopes_, const float p_dropout, const float softmax_scale, const bool is_causal, const int window_size_left, const int window_size_right, const bool return_softmax, const Generator gen_, Tensor* (*allocator)(size_t length)) {
	CATCH(
		auto out = OPTIONAL(out_, at::Tensor);
		auto alibi_slopes = OPTIONAL(alibi_slopes_, at::Tensor);
		auto gen = OPTIONAL(gen_, at::Generator);

		auto res = mha_fwd(*q, *k, *v, out, alibi_slopes, p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, return_softmax, gen);
	
		const size_t sz = res.size();

		Tensor* result = allocator(sz);
		for (size_t i = 0; i < sz; i++)
			result[i] = ResultTensor(res[i]);
	)
}

void THSFlash_MHA_VARLEN_FWD(const Tensor q, const Tensor k, const Tensor v, const Tensor out_, const Tensor cu_seqlens_q, const Tensor cu_seqlens_k, const Tensor seqused_k_, const Tensor alibi_slopes_, const int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors, const bool is_causal, const int window_size_left, const int window_size_right, const bool return_softmax, const Generator gen_, Tensor* (*allocator)(size_t length)) {
	CATCH(
		auto out = OPTIONAL(out_, at::Tensor);
		auto seqused_k = OPTIONAL(seqused_k_, at::Tensor);
		auto alibi_slopes = OPTIONAL(alibi_slopes_, at::Tensor);
		auto gen = OPTIONAL(gen_, at::Generator);
		
		auto res = mha_varlen_fwd(*q, *k, *v, out, *cu_seqlens_q, *cu_seqlens_k, seqused_k, alibi_slopes, max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right, return_softmax, gen);

		const size_t sz = res.size();

		Tensor* result = allocator(sz);
		for (size_t i = 0; i < sz; i++)
			result[i] = ResultTensor(res[i]);
	)
}

void THSFlash_MHA_BWD(const Tensor dout, const Tensor q, const Tensor k, const Tensor v, const Tensor out, const Tensor softmax_lse, const Tensor dq_, const Tensor dk_, const Tensor dv_, const Tensor alibi_slopes_, const float p_dropout, const float softmax_scale, const bool is_causal, const int window_size_left, const int window_size_right, const bool deterministic, const Generator gen_, const Tensor rng_state_, Tensor* (*allocator)(size_t length)) {
	CATCH(
		auto dq = OPTIONAL(dq_, at::Tensor);
		auto dk = OPTIONAL(dk_, at::Tensor);
		auto dv = OPTIONAL(dv_, at::Tensor);
		auto alibi_slopes = OPTIONAL(alibi_slopes_, at::Tensor);
		auto gen = OPTIONAL(gen_, at::Generator);
		auto rng_state = OPTIONAL(rng_state_, at::Tensor);

		auto res = mha_bwd(*dout, *q, *k, *v, *out, *softmax_lse, dq, dk, dv, alibi_slopes, p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, deterministic, gen, rng_state);

		const size_t sz = res.size();

		Tensor * result = allocator(sz);
		for (size_t i = 0; i < sz; i++)
			result[i] = ResultTensor(res[i]);
	)
}

void THSFlash_MHA_VARLEN_BWD(const Tensor dout, const Tensor q, const Tensor k, const Tensor v, const Tensor out, const Tensor softmax_lse, const Tensor dq_, const Tensor dk_, const Tensor dv_, const Tensor cu_seqlens_q, const Tensor cu_seqlens_k, const Tensor alibi_slopes_, const int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors, const bool is_causal, const int window_size_left, const int window_size_right, const bool deterministic, const Generator gen_, const Tensor rng_state_, Tensor* (*allocator)(size_t length)) {
	CATCH(
		auto dq = OPTIONAL(dq_, at::Tensor);
		auto dk = OPTIONAL(dk_, at::Tensor);
		auto dv = OPTIONAL(dv_, at::Tensor);
		auto alibi_slopes = OPTIONAL(alibi_slopes_, at::Tensor);
		auto gen = OPTIONAL(gen_, at::Generator);
		auto rng_state = OPTIONAL(rng_state_, at::Tensor);

		auto res = mha_varlen_bwd(*dout, *q, *k, *v, *out, *softmax_lse, dq, dk, dv, *cu_seqlens_q, *cu_seqlens_k, alibi_slopes, max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right, deterministic, gen, rng_state);

		const size_t sz = res.size();

		Tensor * result = allocator(sz);
		for (size_t i = 0; i < sz; i++)
			result[i] = ResultTensor(res[i]);
	)
}

void THSFlash_MHA_FWD_KVCACHE(const Tensor q, const Tensor kcache, const Tensor vcache, const Tensor k_, const Tensor v_, const Tensor seqlens_k_, const Tensor rotary_cos_, const Tensor rotary_sin_, const Tensor cache_batch_idx_, const Tensor block_table_, const Tensor alibi_slopes_, const Tensor out_, const float softmax_scale, const bool is_causal, const int window_size_left, const int window_size_right, const bool is_rotary_interleaved, const int num_splits, Tensor* (*allocator)(size_t length)) {
	CATCH(
		auto k = OPTIONAL(k_, const at::Tensor);
		auto v = OPTIONAL(v_, const at::Tensor);
		auto seqlens_k = OPTIONAL(seqlens_k_, const at::Tensor);
		auto rotary_cos = OPTIONAL(rotary_cos_, const at::Tensor);
		auto rotary_sin = OPTIONAL(rotary_sin_, const at::Tensor);
		auto cache_batch_idx = OPTIONAL(cache_batch_idx_, const at::Tensor);
		auto block_table = OPTIONAL(block_table_, at::Tensor);
		auto alibi_slopes = OPTIONAL(alibi_slopes_, at::Tensor);
		auto out = OPTIONAL(out_, at::Tensor);
	
		auto res = mha_fwd_kvcache(*q, *kcache, *vcache, k, v, seqlens_k, rotary_cos, rotary_sin, cache_batch_idx, block_table, alibi_slopes, out, softmax_scale, is_causal, window_size_left, window_size_right, is_rotary_interleaved, num_splits);

		const size_t sz = res.size();

		Tensor * result = allocator(sz);
		for (size_t i = 0; i < sz; i++)
			result[i] = ResultTensor(res[i]);
	)
}

char* THSFlash_get_and_reset_last_err()
{
	char* tmp = flash_last_err;
	flash_last_err = nullptr;
	return tmp;
}