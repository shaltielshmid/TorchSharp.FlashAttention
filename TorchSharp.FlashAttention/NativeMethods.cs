using System;
using System.Runtime.InteropServices;

namespace TorchSharp.FlashAttention {
    internal static class NativeMethods { 
        [DllImport("LibFlashAttention")]
        public static extern void THSFlash_MHA_FWD(IntPtr q, IntPtr k, IntPtr v, IntPtr out_, IntPtr alibi_slopes_, float p_dropout, float softmax_scale, [MarshalAs(UnmanagedType.U1)] bool is_causal, int window_size_left, int window_size_right, [MarshalAs(UnmanagedType.U1)] bool return_softmax, IntPtr gen_, AllocatePinnedArray allocator);

        [DllImport("LibFlashAttention")]
        public static extern void THSFlash_MHA_VARLEN_FWD(IntPtr q, IntPtr k, IntPtr v, IntPtr out_, IntPtr cu_seqlens_q, IntPtr cu_seqlens_k, IntPtr seqused_k_, IntPtr alibi_slopes_, int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, [MarshalAs(UnmanagedType.U1)] bool zero_tensors, [MarshalAs(UnmanagedType.U1)] bool is_causal, int window_size_left, int window_size_right, [MarshalAs(UnmanagedType.U1)] bool return_softmax, IntPtr gen_, AllocatePinnedArray allocator);

        [DllImport("LibFlashAttention")]
        public static extern void THSFlash_MHA_BWD(IntPtr dout, IntPtr q, IntPtr k, IntPtr v, IntPtr @out, IntPtr softmax_lse, IntPtr dq_, IntPtr dk_, IntPtr dv_, IntPtr alibi_slopes_, float p_dropout, float softmax_scale, [MarshalAs(UnmanagedType.U1)] bool is_causal, int window_size_left, int window_size_right, [MarshalAs(UnmanagedType.U1)] bool deterministic, IntPtr gen_, IntPtr rng_state_, AllocatePinnedArray allocator);

        [DllImport("LibFlashAttention")]
        public static extern void THSFlash_MHA_VARLEN_BWD(IntPtr dout, IntPtr q, IntPtr k, IntPtr v, IntPtr @out, IntPtr softmax_lse, IntPtr dq_, IntPtr dk_, IntPtr dv_, IntPtr cu_seqlens_q, IntPtr cu_seqlens_k, IntPtr alibi_slopes_, int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, [MarshalAs(UnmanagedType.U1)] bool zero_tensors, [MarshalAs(UnmanagedType.U1)] bool is_causal, int window_size_left, int window_size_right, [MarshalAs(UnmanagedType.U1)] bool deterministic, IntPtr gen_, IntPtr rng_state_, AllocatePinnedArray allocator);

        [DllImport("LibFlashAttention")]
        public static extern void THSFlash_MHA_FWD_KVCACHE(IntPtr q, IntPtr kcache, IntPtr vcache, IntPtr k_, IntPtr v_, IntPtr seqlens_k_, IntPtr rotary_cos_, IntPtr rotary_sin_, IntPtr cache_batch_idx_, IntPtr block_table_, IntPtr alibi_slopes_, IntPtr out_, float softmax_scale, [MarshalAs(UnmanagedType.U1)] bool is_causal, int window_size_left, int window_size_right, [MarshalAs(UnmanagedType.U1)] bool is_rotary_interleaved, int num_splits, AllocatePinnedArray allocator);

        [DllImport("LibFlashAttention")]
        internal static extern IntPtr THSFlash_get_and_reset_last_err();
    }
}

