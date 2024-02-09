using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace TorchSharp.FlashAttention.FlashAttentionFunctions
{
    internal static class Utils {

        internal static List<torch.Tensor> FlashAttentionForward(torch.Tensor q, torch.Tensor k, torch.Tensor v, float dropout_p, float softmax_scale, bool causal, (int left, int right) window_size, torch.Tensor? alibi_slopes, bool return_softmax) {
            q = MaybeContiguous(q);
            k = MaybeContiguous(k);
            v = MaybeContiguous(v);

            var results = new PinnedArray<IntPtr>();

            NativeMethods.THSFlash_MHA_FWD(handle(q), handle(k), handle(v), IntPtr.Zero, handle(alibi_slopes), dropout_p, softmax_scale, causal, window_size.left, window_size.right, return_softmax, IntPtr.Zero, results.CreateArray);
            CheckForErrors();

            return results.Array!.Select(Tensor.UnsafeCreateTensor).ToList();
        }

        internal static List<torch.Tensor> FlashAttentionVarLenForward(torch.Tensor q, torch.Tensor k, torch.Tensor v, torch.Tensor cu_seqlens_q, torch.Tensor cu_seqlens_k, int max_seqlen_q, int max_seqlen_k, float dropout_p, float softmax_scale, bool causal, (int left, int right) window_size, torch.Tensor? alibi_slopes, bool return_softmax) {
            q = MaybeContiguous(q);
            k = MaybeContiguous(k);
            v = MaybeContiguous(v);

            var results = new PinnedArray<IntPtr>();

            NativeMethods.THSFlash_MHA_VARLEN_FWD(handle(q), handle(k), handle(v), IntPtr.Zero, handle(cu_seqlens_q), handle(cu_seqlens_k), IntPtr.Zero, handle(alibi_slopes), max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, false, causal, window_size.left, window_size.right, return_softmax, IntPtr.Zero, results.CreateArray);
            CheckForErrors();

            return results.Array!.Select(Tensor.UnsafeCreateTensor).ToList();
        }

        internal static List<torch.Tensor> FlashAttentionBackward(torch.Tensor dout, torch.Tensor q, torch.Tensor k, torch.Tensor v, torch.Tensor @out, torch.Tensor softmax_lse, torch.Tensor dq, torch.Tensor dk, torch.Tensor dv, float dropout_p, float softmax_scale, bool causal, (int left, int right) window_size, torch.Tensor? alibi_slopes, bool deterministic, torch.Tensor? rng_state = null) {
            // dq, dk, dv are allocated by us so they should already be contiguous
            dout = MaybeContiguous(dout);
            q = MaybeContiguous(q);
            k = MaybeContiguous(k);
            v = MaybeContiguous(v);
            @out = MaybeContiguous(@out);

            var results = new PinnedArray<IntPtr>();

            NativeMethods.THSFlash_MHA_BWD(handle(dout), handle(q), handle(k), handle(v), handle(@out), handle(softmax_lse), handle(dq), handle(dk), handle(dv), handle(alibi_slopes), dropout_p, softmax_scale, causal, window_size.left, window_size.right, deterministic, IntPtr.Zero, handle(rng_state), results.CreateArray);
            CheckForErrors();

            return results.Array!.Select(Tensor.UnsafeCreateTensor).ToList();

        }

        internal static List<torch.Tensor> FlashAttentionVarLenBackward(torch.Tensor dout, torch.Tensor q, torch.Tensor k, torch.Tensor v, torch.Tensor @out, torch.Tensor softmax_lse, torch.Tensor dq, torch.Tensor dk, torch.Tensor dv, torch.Tensor cu_seqlens_q, torch.Tensor cu_seqlens_k, int max_seqlen_q, int max_seqlen_k, float dropout_p, float softmax_scale, bool causal, (int left, int right) window_size, torch.Tensor? alibi_slopes, bool deterministic, torch.Tensor? rng_state = null) {
            // dq, dk, dv are allocated by us so they should already be contiguous
            dout = MaybeContiguous(dout);
            q = MaybeContiguous(q);
            k = MaybeContiguous(k);
            v = MaybeContiguous(v);
            @out = MaybeContiguous(@out);

            var results = new PinnedArray<IntPtr>();

            NativeMethods.THSFlash_MHA_VARLEN_BWD(handle(dout), handle(q), handle(k), handle(v), handle(@out), handle(softmax_lse), handle(dq), handle(dk), handle(dv), handle(cu_seqlens_q), handle(cu_seqlens_k), handle(alibi_slopes), max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, false, causal, window_size.left, window_size.right, deterministic, IntPtr.Zero, handle(rng_state), results.CreateArray);
            CheckForErrors();

            return results.Array!.Select(Tensor.UnsafeCreateTensor).ToList();
        }

        public static List<torch.Tensor> FlashAttentionForwardKVCache(torch.Tensor q, torch.Tensor k_cache, torch.Tensor v_cache, torch.Tensor? k, torch.Tensor? v, torch.Tensor? rotary_cos, torch.Tensor? rotary_sin, torch.Tensor? cache_seqlens, torch.Tensor? cache_batch_idx, torch.Tensor? block_table, float softmax_scale, bool causal, (int left, int right) window_size, bool rotary_interleaved, torch.Tensor? alibi_slopes, int num_splits) {
            if (k_cache.stride(-1) != 1) throw new ArgumentException("k_cache must have contiguous last dimension", nameof(k_cache));
            if (v_cache.stride(-1) != 1) throw new ArgumentException("v_cache must have contiguous last dimension", nameof(v_cache));

            q = MaybeContiguous(q);
            k = k is null ? k : MaybeContiguous(k);
            v = v is null ? v : MaybeContiguous(v);
            cache_seqlens = cache_seqlens is null ? cache_seqlens : MaybeContiguous(cache_seqlens);
            cache_batch_idx = cache_batch_idx is null ? cache_batch_idx : MaybeContiguous(cache_batch_idx);
            block_table = block_table is null ? block_table : MaybeContiguous(block_table);

            var results = new PinnedArray<IntPtr>();

            NativeMethods.THSFlash_MHA_FWD_KVCACHE(handle(q), handle(k_cache), handle(v_cache), handle(k), handle(v), handle(cache_seqlens), handle(rotary_cos), handle(rotary_sin), handle(cache_batch_idx), handle(block_table), handle(alibi_slopes), IntPtr.Zero, softmax_scale, causal, window_size.left, window_size.right, rotary_interleaved, num_splits, results.CreateArray);
            CheckForErrors();

            return results.Array!.Select(Tensor.UnsafeCreateTensor).ToList();
        }

        internal static void CheckForErrors() {
            var error = NativeMethods.THSFlash_get_and_reset_last_err();

            if (error != IntPtr.Zero) {
                throw new ExternalException(Marshal.PtrToStringAnsi(error));
            }
        }

        private static torch.Tensor MaybeContiguous(torch.Tensor x) => x.stride(-1) != 1 ? x.contiguous() : x;

        private static IntPtr handle(torch.Tensor? x) => x is null || x.IsInvalid ? IntPtr.Zero : x.Handle;

        private static Tensor CreateTensor(IntPtr handle) {
            return (Tensor)Activator.CreateInstance(typeof(Tensor), handle)!;
        }
    }
}