﻿using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace TorchSharp.FlashAttention.FlashAttentionFunctions {
    internal class FlashAttnVarlenQKVPackedFunc : torch.autograd.MultiTensorFunction<FlashAttnVarlenQKVPackedFunc> {
        public override string Name => nameof(FlashAttnVarlenQKVPackedFunc);

        public override List<torch.Tensor> backward(torch.autograd.AutogradContext ctx, List<torch.Tensor> grad_outputs) {
            var dout = grad_outputs[0];

            var saved = ctx.get_saved_variables();
            var (q, k, v, @out, softmax_lse, cu_seqlens, rng_state, alibi_slopes) = (saved[0], saved[1], saved[2], saved[3], saved[4], saved[5], saved[6], saved[7]);
            long[] qkv_shape = [.. q.shape[..^2], 3, .. q.shape[^2..]];

            var dqkv = torch.empty(qkv_shape, q.dtype, q.device);

            Utils.FlashAttentionVarLenBackward(
                    dout, q, k, v, @out, softmax_lse, dqkv[.., .., 0], dqkv[.., .., 1], dqkv[.., .., 2],
                    cu_seqlens, cu_seqlens, (int)ctx.get_data("max_seqlen"), (int)ctx.get_data("max_seqlen"),
                    (float)ctx.get_data("dropout_p"), (float)ctx.get_data("softmax_scale"), (bool)ctx.get_data("causal"), 
                    ((int, int))ctx.get_data("window_size"), alibi_slopes, (bool)ctx.get_data("deterministic"), rng_state);

            dqkv = dqkv.slice(-1, 0, dout.shape[^1], 1);
            return [dqkv, null, null];
        }

        public override List<torch.Tensor> forward(torch.autograd.AutogradContext ctx, params object[] vars) {
            var qkv = (torch.Tensor)vars[0];
            var cu_seqlens = (torch.Tensor)vars[1];
            int max_seqlen = (int)vars[2];
            float dropout_p = (float)vars[3];
            float softmax_scale = (float?)vars[4] ?? MathF.Pow(qkv.shape[^1], -0.5f);
            bool causal = (bool)vars[5];
            var window_size = ((int left, int right))vars[6];
            var alibi_slopes = (torch.Tensor?)vars[7];
            bool deterministic = (bool)vars[8];
            bool return_softmax = (bool)vars[9];

            var res = Utils.FlashAttentionVarLenForward(
                    qkv[.., .., 0], qkv[.., .., 1], qkv[.., .., 2], cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                    dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax);
            // res = [out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state]
            var (@out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state) = (res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);

            ctx.save_for_backward([q, k, v, out_padded, softmax_lse, cu_seqlens, rng_state, alibi_slopes]);
            ctx.save_data("dropout_p", dropout_p);
            ctx.save_data("max_seqlen", max_seqlen);
            ctx.save_data("softmax_scale", softmax_scale);
            ctx.save_data("causal", causal);
            ctx.save_data("window_size", window_size);
            ctx.save_data("deterministic", deterministic);

            return return_softmax ? [out_padded, softmax_lse, S_dmask] : [@out];
        }
    }
}