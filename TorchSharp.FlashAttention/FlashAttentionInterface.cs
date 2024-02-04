using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.FlashAttention.FlashAttentionFunctions;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TorchSharp.FlashAttention {
    public static class FlashAttentionInterface {

        /// <summary>
        /// dropout_p should be set to 0.0 during evaluation
        /// If Q, K, V are already stacked into 1 tensor, this function will be faster than
        /// calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
        /// of the gradients of Q, K, V.
        /// For multi-query and grouped-query attention (MQA/GQA), please see
        /// flash_attn_kvpacked_func and flash_attn_func.
        ///
        /// If window_size != null or (-1, -1), implements sliding window local attention. Query at position i
        /// will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.
        /// </summary>
        /// <param name="qkv">(batch_size, seqlen, 3, nheads, headdim)</param>
        /// <param name="dropout_p">float. Dropout probability.</param>
        /// <param name="softmax_scale">float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).</param>
        /// <param name="causal">bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).</param>
        /// <param name="window_size">(left, right). If not null or (-1, -1), implements sliding window local attention.</param>
        /// <param name="alibi_slopes">(nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to the attention score of query i and key j.</param>
        /// <param name="deterministic">bool. Whether to use the deterministic implementation of the backward pass, which is slightly slower and uses more memory. The forward pass is always deterministic.</param>
        /// <param name="return_attn_probs">bool. Whether to return the attention probabilities. This option is for testing only. The returned probabilities are not guaranteed to be correct (they might not have the right scaling).</param>
        /// <returns>
        /// out: (batch_size, seqlen, nheads, headdim).
        /// softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        /// S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen). The output of softmax (possibly with different scaling). It also encodes the dropout pattern (negative means that location was dropped, nonnegative means it was kept).
        /// </returns>
        public static (torch.Tensor @out, torch.Tensor? softmax_lse, torch.Tensor? S_dmask) flash_attn_qkvpacked_func(torch.Tensor qkv, float dropout_p = 0f, float? softmax_scale = null, bool causal = false, (int left, int right)? window_size = null, torch.Tensor? alibi_slopes = null, bool deterministic = false, bool return_attn_probs = false) {
            var ret = FlashAttnQKVPackedFunc.apply(qkv, dropout_p, softmax_scale, causal, window_size ?? (-1, -1), alibi_slopes, deterministic, return_attn_probs);
            return ret.Count == 1 ? (ret[0], null, null) : (ret[0], ret[1], ret[2]);
        }

        /// <summary>
        /// dropout_p should be set to 0.0 during evaluation
        /// If K, V are already stacked into 1 tensor, this function will be faster than
        /// calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
        /// of the gradients of K, V.
        /// Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
        /// than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        /// For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        /// 0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
        /// 
        /// If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        /// For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        ///     1 1 1 1 0
        ///     1 1 1 1 1
        /// If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        ///     0 0
        ///     0 0
        ///     0 0
        ///     1 0
        ///     1 1
        /// If the row of the mask is all zero, the output will be zero.
        /// 
        /// If window_size != (-1, -1) or null, implements sliding window local attention. Query at position i
        /// will only attend to keys between
        /// [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
        /// </summary>
        /// <param name="q">(batch_size, seqlen, nheads, headdim)</param>
        /// <param name="kv">(batch_size, seqlen, 2, nheads_k, headdim)</param>
        /// <param name="dropout_p">float. Dropout probability.</param>
        /// <param name="softmax_scale">float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).</param>
        /// <param name="causal">bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).</param>
        /// <param name="window_size">(left, right). If not (-1, -1) or null, implements sliding window local attention.</param>
        /// <param name="alibi_slopes">(nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i + seqlen_k - seqlen_q - j|) is added to the attention score of query i and key j.</param>
        /// <param name="deterministic">bool. Whether to use the deterministic implementation of the backward pass, which is slightly slower and uses more memory. The forward pass is always deterministic.</param>
        /// <param name="return_attn_probs">bool. Whether to return the attention probabilities. This option is for testing only. The returned probabilities are not guaranteed to be correct (they might not have the right scaling).</param>
        /// <returns>
        /// out: (batch_size, seqlen, nheads, headdim).
        /// softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        /// S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen). The output of softmax (possibly with different scaling). It also encodes the dropout pattern (negative means that location was dropped, nonnegative means it was kept).
        /// </returns>
        public static (torch.Tensor @out, torch.Tensor? softmax_lse, torch.Tensor? S_dmask) flash_attn_kvpacked_func(torch.Tensor q, torch.Tensor kv, float dropout_p = 0, float? softmax_scale = null, bool causal = false, (int left, int right)? window_size = null, torch.Tensor? alibi_slopes = null, bool deterministic = false, bool return_attn_probs = false) {
            var ret = FlashAttnKVPackedFunc.apply(q, kv, dropout_p, softmax_scale, causal, window_size ?? (-1, -1), alibi_slopes, deterministic, return_attn_probs);
            return (ret[0], ret[1], ret[2]);
        }

        /// <summary>
        /// dropout_p should be set to 0.0 during evaluation
        /// Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
        /// than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        /// For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        /// 0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
        /// 
        /// If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        /// For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        ///     1 1 1 1 0
        ///     1 1 1 1 1
        /// If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        ///     0 0
        ///     0 0
        ///     0 0
        ///     1 0
        ///     1 1
        /// If the row of the mask is all zero, the output will be zero.
        /// 
        /// If window_size != (-1, -1) or null, implements sliding window local attention. Query at position i
        /// will only attend to keys between
        /// [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
        /// </summary>
        /// <param name="q">(batch_size, seqlen, nheads, headdim)</param>
        /// <param name="k">(batch_size, seqlen, nheads_k, headdim)</param>
        /// <param name="v">(batch_size, seqlen, nheads_k, headdim)</param>
        /// <param name="dropout_p">float. Dropout probability.</param>
        /// <param name="softmax_scale">float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).</param>
        /// <param name="causal">bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).</param>
        /// <param name="window_size">(left, right). If not null or (-1, -1), implements sliding window local attention.</param>
        /// <param name="alibi_slopes">(nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to the attention score of query i and key j.</param>
        /// <param name="deterministic">bool. Whether to use the deterministic implementation of the backward pass, which is slightly slower and uses more memory. The forward pass is always deterministic.</param>
        /// <param name="return_attn_probs">bool. Whether to return the attention probabilities. This option is for testing only. The returned probabilities are not guaranteed to be correct (they might not have the right scaling).</param>
        /// <returns>
        /// out: (batch_size, seqlen, nheads, headdim).
        /// softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        /// S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen). The output of softmax (possibly with different scaling). It also encodes the dropout pattern (negative means that location was dropped, nonnegative means it was kept).
        /// </returns>
        public static (torch.Tensor @out, torch.Tensor? softmax_lse, torch.Tensor? S_dmask) flash_attn_func(torch.Tensor q, torch.Tensor k, torch.Tensor v, float dropout_p = 0, float? softmax_scale = null, bool causal = false, (int left, int right)? window_size = null, torch.Tensor? alibi_slopes = null, bool deterministic = false, bool return_attn_probs = false) {
            var ret = FlashAttnFunc.apply(q, k, v, dropout_p, softmax_scale, causal, window_size ?? (-1, -1), alibi_slopes, deterministic, return_attn_probs);
            return (ret[0], ret[1], ret[2]);
        }


        /// <summary>
        /// dropout_p should be set to 0.0 during evaluation
        /// If Q, K, V are already stacked into 1 tensor, this function will be faster than
        /// calling flash_attn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
        /// of the gradients of Q, K, V.
        /// For multi-query and grouped-query attention (MQA/GQA), please see
        /// flash_attn_varlen_kvpacked_func and flash_attn_varlen_func.
        /// 
        /// If window_size != (-1, -1) or null, implements sliding window local attention. Query at position i
        /// will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.
        /// </summary>
        /// <param name="qkv">(total, 3, nheads, headdim), where total = total number of tokens in the batch.</param>
        /// <param name="cu_seqlens">(batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the sequences in the batch, used to index into qkv.</param>
        /// <param name="max_seqlen">int. Maximum sequence length in the batch.</param>
        /// <param name="dropout_p">float. Dropout probability.</param>
        /// <param name="softmax_scale">float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).</param>
        /// <param name="causal">bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).</param>
        /// <param name="window_size">(left, right). If not null or (-1, -1), implements sliding window local attention.</param>
        /// <param name="alibi_slopes">(nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to the attention score of query i and key j.</param>
        /// <param name="deterministic">bool. Whether to use the deterministic implementation of the backward pass, which is slightly slower and uses more memory. The forward pass is always deterministic.</param>
        /// <param name="return_attn_probs">bool. Whether to return the attention probabilities. This option is for testing only. The returned probabilities are not guaranteed to be correct (they might not have the right scaling).</param>
        /// <returns>
        /// out: (batch_size, seqlen, nheads, headdim).
        /// softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        /// S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen). The output of softmax (possibly with different scaling). It also encodes the dropout pattern (negative means that location was dropped, nonnegative means it was kept).
        /// </returns>
        public static (torch.Tensor @out, torch.Tensor? softmax_lse, torch.Tensor? S_dmask) flash_attn_varlen_qkvpacked_func(torch.Tensor qkv, torch.Tensor cu_seqlens, int max_seqlen, float dropout_p = 0, float? softmax_scale = null, bool causal = false, (int left, int right)? window_size = null, torch.Tensor? alibi_slopes = null, bool deterministic = false, bool return_attn_probs = false) {
            var ret = FlashAttnVarlenQKVPackedFunc.apply(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal, window_size ?? (-1, -1), alibi_slopes, deterministic, return_attn_probs);
            return ret.Count == 1 ? (ret[0], null, null) : (ret[0], ret[1], ret[2]);
        }

        /// <summary>
        /// dropout_p should be set to 0.0 during evaluation
        /// If K, V are already stacked into 1 tensor, this function will be faster than
        /// calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
        /// of the gradients of K, V.
        /// Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
        /// than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        /// For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        /// 0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
        /// 
        /// If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        /// For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        ///     1 1 1 1 0
        ///     1 1 1 1 1
        /// If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        ///     0 0
        ///     0 0
        ///     0 0
        ///     1 0
        ///     1 1
        /// If the row of the mask is all zero, the output will be zero.
        /// 
        /// If window_size != (-1, -1), implements sliding window local attention. Query at position i
        /// will only attend to keys between
        /// [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
        /// </summary>
        /// <param name="q">(total_q, nheads, headdim), where total_q = total number of query tokens in the batch.</param>
        /// <param name="kv">(total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.</param>
        /// <param name="cu_seqlens_q">(batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the sequences in the batch, used to index into q.</param>
        /// <param name="cu_seqlen_k">(batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the sequences in the batch, used to index into kv.</param>
        /// <param name="max_seqlen_q">int. Maximum query sequence length in the batch.</param>
        /// <param name="max_seqlen_k">int. Maximum key sequence length in the batch.</param>
        /// <param name="dropout_p">float. Dropout probability.</param>
        /// <param name="softmax_scale">float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).</param>
        /// <param name="causal">bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).</param>
        /// <param name="window_size">(left, right). If not null or (-1, -1), implements sliding window local attention.</param>
        /// <param name="alibi_slopes">(nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to the attention score of query i and key j.</param>
        /// <param name="deterministic">bool. Whether to use the deterministic implementation of the backward pass, which is slightly slower and uses more memory. The forward pass is always deterministic.</param>
        /// <param name="return_attn_probs">bool. Whether to return the attention probabilities. This option is for testing only. The returned probabilities are not guaranteed to be correct (they might not have the right scaling).</param>
        /// <returns>
        /// out: (batch_size, seqlen, nheads, headdim).
        /// softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        /// S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen). The output of softmax (possibly with different scaling). It also encodes the dropout pattern (negative means that location was dropped, nonnegative means it was kept).
        /// </returns>
        public static (torch.Tensor @out, torch.Tensor? softmax_lse, torch.Tensor? S_dmask) flash_attn_varlen_kvpacked_func(torch.Tensor q, torch.Tensor kv, torch.Tensor cu_seqlens_q, torch.Tensor cu_seqlen_k, int max_seqlen_q, int max_seqlen_k, float dropout_p = 0f, float? softmax_scale = null, bool causal = false, (int left, int right)? window_size = null, torch.Tensor? alibi_slopes = null, bool deterministic = false, bool return_attn_probs = false) {
            var ret = FlashAttnVarlenKVPackedFunc.apply(q, kv, cu_seqlens_q, cu_seqlen_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size ?? (-1, -1), alibi_slopes, deterministic, return_attn_probs);
            return (ret[0], ret[1], ret[2]);
        }


        /// <summary>
        /// dropout_p should be set to 0.0 during evaluation
        /// Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
        /// than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        /// For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        /// 0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
        /// 
        /// If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        /// For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        ///     1 1 1 1 0
        ///     1 1 1 1 1
        /// If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        ///     0 0
        ///     0 0
        ///     0 0
        ///     1 0
        ///     1 1
        /// If the row of the mask is all zero, the output will be zero.
        /// 
        /// If window_size != (-1, -1), implements sliding window local attention. Query at position i
        /// will only attend to keys between
        /// [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
        /// </summary>
        /// <param name="q">(total_q, nheads, headdim), where total_q = total number of query tokens in the batch.</param>
        /// <param name="k">(total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.</param>
        /// <param name="v">(total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.</param>
        /// <param name="cu_seqlens_q">(batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the sequences in the batch, used to index into q.</param>
        /// <param name="cu_seqlen_k">(batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the sequences in the batch, used to index into kv.</param>
        /// <param name="max_seqlen_q">int. Maximum query sequence length in the batch.</param>
        /// <param name="max_seqlen_k">int. Maximum key sequence length in the batch.</param>
        /// <param name="dropout_p">float. Dropout probability.</param>
        /// <param name="softmax_scale">float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).</param>
        /// <param name="causal">bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).</param>
        /// <param name="window_size">(left, right). If not null or (-1, -1), implements sliding window local attention.</param>
        /// <param name="alibi_slopes">(nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to the attention score of query i and key j.</param>
        /// <param name="deterministic">bool. Whether to use the deterministic implementation of the backward pass, which is slightly slower and uses more memory. The forward pass is always deterministic.</param>
        /// <param name="return_attn_probs">bool. Whether to return the attention probabilities. This option is for testing only. The returned probabilities are not guaranteed to be correct (they might not have the right scaling).</param>
        /// <returns>
        /// out: (batch_size, seqlen, nheads, headdim).
        /// softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        /// S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen). The output of softmax (possibly with different scaling). It also encodes the dropout pattern (negative means that location was dropped, nonnegative means it was kept).
        /// </returns>
        public static (torch.Tensor @out, torch.Tensor? softmax_lse, torch.Tensor? S_dmask) flash_attn_varlen_func(torch.Tensor q, torch.Tensor k, torch.Tensor v, torch.Tensor cu_seqlens_q, torch.Tensor cu_seqlen_k, int max_seqlen_q, int max_seqlen_k, float dropout_p = 0f, float? softmax_scale = null, bool causal = false, (int left, int right)? window_size = null, torch.Tensor? alibi_slopes = null, bool deterministic = false, bool return_attn_probs = false) {
            var ret = FlashAttnVarlenFunc.apply(q, k, v, cu_seqlens_q, cu_seqlen_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size ?? (-1, -1), alibi_slopes, deterministic, return_attn_probs);
            return (ret[0], ret[1], ret[2]);
        }

        /// <summary>
        /// If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
        /// k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
        /// the previous step, and update them with the new keys/values from the current step, and do
        /// attention with the updated cache, all in 1 kernel.
        /// 
        /// If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
        /// For example, the KV cache could be pre-allocated with the max sequence length, and you can use
        /// cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.
        /// 
        /// Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
        /// rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
        /// If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
        /// and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
        /// If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
        /// indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).
        /// 
        /// See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.
        /// 
        /// Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
        /// than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        /// For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        /// 0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
        /// 
        /// If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        /// For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        ///     1 1 1 1 0
        ///     1 1 1 1 1
        /// If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        ///     0 0
        ///     0 0
        ///     0 0
        ///     1 0
        ///     1 1
        /// If the row of the mask is all zero, the output will be zero.
        /// 
        /// If window_size != (-1, -1), implements sliding window local attention. Query at position i
        /// will only attend to keys between
        /// [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
        /// 
        /// Note: Does not support backward pass.
        /// </summary>
        /// <param name="q">(batch_size, seqlen, nheads, headdim)</param>
        /// <param name="k_cache">(batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table, or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache) page_block_size must be a multiple of 256.</param>
        /// <param name="v_cache">(batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table, or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)</param>
        /// <param name="k">(batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate k with k_cache, starting at the indices specified by cache_seqlens.</param>
        /// <param name="v">(batch_size, seqlen_new, nheads_k, headdim). Similar to k.</param>
        /// <param name="rotary_cos">(seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.</param>
        /// <param name="rotary_sin">(seqlen_ro, rotary_dim / 2). Similar to rotary_cos.</param>
        /// <param name="cache_seqlens">int or (batch_size,), dtype torch.int32. The sequence lengths of the KV cache.</param>
        /// <param name="cache_batch_idx">(batch_size,), dtype torch.int32. The indices used to index into the KV cache. If None, we assume that the batch indices are[0, 1, 2, ..., batch_size - 1]. If the indices are not distinct, and k and v are provided, the values updated in the cache might come from any of the duplicate indices.</param>
        /// <param name="block_table">(batch_size, max_num_blocks_per_seq), dtype torch.int32.</param>
        /// <param name="softmax_scale">float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).</param>
        /// <param name="causal">bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).</param>
        /// <param name="window_size">(left, right). If not (-1, -1), implements sliding window local attention.</param>
        /// <param name="rotary_interleaved">bool. Only applicable if rotary_cos and rotary_sin are passed in. If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False, rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1 (i.e.GPT-NeoX style).</param>
        /// <param name="alibi_slopes">(nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i + seqlen_k - seqlen_q - j|) is added to the attention score of query i and key j.</param>
        /// <param name="num_splits">int. If > 1, split the key/value into this many chunks along the sequence. If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic to automatically determine the number of splits. Don't change this unless you know what you are doing.</param>
        /// <returns>out: (batch_size, seqlen, nheads, headdim).</returns>
        public static torch.Tensor flash_attn_with_kvcache(torch.Tensor q, torch.Tensor k_cache, torch.Tensor v_cache, torch.Tensor? k = null, torch.Tensor? v = null, torch.Tensor? rotary_cos = null, torch.Tensor? rotary_sin = null, torch.Tensor? cache_seqlens = null, torch.Tensor? cache_batch_idx = null, torch.Tensor? block_table = null, float? softmax_scale = null, bool causal = false, (int left, int right)? window_size = null, bool rotary_interleaved = true, torch.Tensor? alibi_slopes = null, int num_splits = 0) {
            if (cache_seqlens is not null && cache_seqlens.numel() == 1) 
                cache_seqlens = torch.full(1, k_cache.shape[0], cache_seqlens.item<int>(), torch.int32, k_cache.device);

            var ret = FlashAttentionFunctions.Utils.FlashAttentionForwardKVCache(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens, cache_batch_idx, block_table, softmax_scale ?? MathF.Pow(q.shape[^1], -0.5f), causal, window_size ?? (-1, -1), rotary_interleaved, alibi_slopes, num_splits);
            // ret = [out, softmax_lse]

            return ret[0];
        }
    }


}