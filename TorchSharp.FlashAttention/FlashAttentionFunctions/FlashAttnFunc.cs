﻿using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.FlashAttention.BertPaddingFunctions;
using static TorchSharp.torch;

namespace TorchSharp.FlashAttention.FlashAttentionFunctions {
    internal class FlashAttnFunc : torch.autograd.MultiTensorFunction<FlashAttnFunc> {
        public override string Name => nameof(FlashAttnFunc);

        public override List<torch.Tensor?> backward(torch.autograd.AutogradContext ctx, List<torch.Tensor> grad_outputs) {
            var dout = grad_outputs[0];

            var saved = ctx.get_saved_variables();
            var (q, k, v, @out, softmax_lse, rng_state, alibi_slopes) = (saved[0], saved[1], saved[2], saved[3], saved[4], saved[5], saved[6]);
            
            var dq = torch.empty_like(q);
            var dk = torch.empty_like(k);
            var dv = torch.empty_like(v);
            
            Utils.FlashAttentionBackward(
                    dout, q, k, v, @out, softmax_lse, dq, dk, dv,
                    (float)ctx.get_data("dropout_p"), (float)ctx.get_data("softmax_scale"), (bool)ctx.get_data("causal"), 
                    ((int, int))ctx.get_data("window_size"), alibi_slopes, (bool)ctx.get_data("deterministic"), rng_state);

            dq = dq.slice(-1, 0, dout.shape[^1], 1);
            dk = dk.slice(-1, 0, dout.shape[^1], 1);
            dv = dv.slice(-1, 0, dout.shape[^1], 1);
            return new() { dq, dk, dv, null, null, null, null, null, null, null };
        }

        public override List<torch.Tensor> forward(torch.autograd.AutogradContext ctx, params object[] vars) {
            var q_input = (torch.Tensor)vars[0];
            var k_input = (torch.Tensor)vars[1];
            var v_input = (torch.Tensor)vars[2];
            float dropout_p = (float)vars[3];
            float softmax_scale = (float?)vars[4] ?? MathF.Pow(q_input.shape[^1], -0.5f);
            bool causal = (bool)vars[5];
            var window_size = ((int left, int right))vars[6];
            var alibi_slopes = (torch.Tensor?)vars[7];
            bool deterministic = (bool)vars[8];
            bool return_softmax = (bool)vars[9];

            var res = Utils.FlashAttentionForward(q_input, k_input, v_input, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax);
            // res = [out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state]
            var (@out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state) = (res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);

            ctx.save_for_backward(new() { q, k, v, out_padded, softmax_lse, rng_state, alibi_slopes });
            ctx.save_data("dropout_p", dropout_p);
            ctx.save_data("softmax_scale", softmax_scale);
            ctx.save_data("causal", causal);
            ctx.save_data("window_size", window_size);
            ctx.save_data("deterministic", deterministic);

            return return_softmax ? new() { out_padded, softmax_lse, S_dmask } : new() { @out };
        }
    }
}
