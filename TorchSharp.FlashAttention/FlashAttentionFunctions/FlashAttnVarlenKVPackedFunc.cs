using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace TorchSharp.FlashAttention.FlashAttentionFunctions {
    internal class FlashAttnVarlenKVPackedFunc : torch.autograd.MultiTensorFunction<FlashAttnVarlenKVPackedFunc> {
        public override string Name => nameof(FlashAttnVarlenKVPackedFunc);

        public override List<torch.Tensor> backward(torch.autograd.AutogradContext ctx, List<torch.Tensor> grad_outputs) {
            var dout = grad_outputs[0];

            var saved = ctx.get_saved_variables();
            var (q, k, v, @out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, alibi_slopes) = (saved[0], saved[1], saved[2], saved[3], saved[4], saved[5], saved[6], saved[7], saved[8]);

            var dq = torch.empty_like(q);
            long[] kv_shape = k.shape[..^2].Append(2).Concat(k.shape[^2..]).ToArray();
            var dkv = torch.empty(kv_shape, k.dtype, k.device);

            Utils.FlashAttentionVarLenBackward(
                    dout, q, k, v, @out, softmax_lse, dq, dkv[.., .., 0], dkv[.., .., 1],
                    cu_seqlens_q, cu_seqlens_k, (int)ctx.get_data("max_seqlen_q"), (int)ctx.get_data("max_seqlen_k"),
                    (float)ctx.get_data("dropout_p"), (float)ctx.get_data("softmax_scale"), (bool)ctx.get_data("causal"), 
                    ((int, int))ctx.get_data("window_size"), alibi_slopes, (bool)ctx.get_data("deterministic"), rng_state);

            dq = dq.slice(-1, 0, dout.shape[^1], 1);
            dkv = dkv.slice(-1, 0, dout.shape[^1], 1);
            return new() { dq, dkv, null };
        }

        public override List<torch.Tensor> forward(torch.autograd.AutogradContext ctx, params object[] vars) {
            var q_input = (torch.Tensor)vars[0];
            var kv = (torch.Tensor)vars[1];
            var cu_seqlens_q = (torch.Tensor)vars[2];
            var cu_seqlens_k = (torch.Tensor)vars[3];
            int max_seqlen_q = (int)vars[4];
            int max_seqlen_k = (int)vars[5];
            float dropout_p = (float)vars[6];
            float softmax_scale = (float?)vars[7] ?? MathF.Pow(q_input.shape[^1], -0.5f);
            bool causal = (bool)vars[8];
            var window_size = ((int left, int right))vars[9];
            var alibi_slopes = (torch.Tensor?)vars[10];
            bool deterministic = (bool)vars[11];
            bool return_softmax = (bool)vars[12];

            var res = Utils.FlashAttentionVarLenForward(
                    q_input, kv[.., .., 0], kv[.., .., 1], cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax);
            // res = [out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state]
            var (@out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state) = (res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);

            ctx.save_for_backward(new() { q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, alibi_slopes });
            ctx.save_data("dropout_p", dropout_p);
            ctx.save_data("max_seqlen_q", max_seqlen_q);
            ctx.save_data("max_seqlen_k", max_seqlen_k);
            ctx.save_data("softmax_scale", softmax_scale);
            ctx.save_data("causal", causal);
            ctx.save_data("window_size", window_size);
            ctx.save_data("deterministic", deterministic);

            return return_softmax ? new() { out_padded, softmax_lse, S_dmask } : new() { @out, null, null };
        }
    }
}
