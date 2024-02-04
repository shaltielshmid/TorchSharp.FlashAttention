using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.FlashAttention {
    public class FlashAttention : torch.nn.Module<torch.Tensor, torch.Tensor?, torch.Tensor> {
        private readonly float? _softmax_scale;
        private readonly float _attention_dropout;
        private readonly bool _causal;

        public FlashAttention(float? softmax_scale, float attention_dropout, bool causal) : base(nameof(FlashAttention)) {
            _softmax_scale = softmax_scale;
            _attention_dropout = attention_dropout;  
            _causal = causal;
        }

        /// <summary>
        /// Implements the multihead softmax attention.
        /// </summary>
        /// <param name="qkv">The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None. if unpadded: (nnz, 3, h, d)</param>
        /// <param name="key_padding_mask">a bool tensor of shape(B, S)</param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public override torch.Tensor forward(torch.Tensor qkv, torch.Tensor? key_padding_mask = null) {

            long batch_size = qkv.shape[0];
            long seqlen = qkv.shape[1];

            if (key_padding_mask is null) {
                qkv = qkv.view([-1, .. qkv.shape[2..]]);
                int max_s = (int)seqlen;
                var cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step: seqlen, dtype: torch.int32, device: qkv.device);

                var output = FlashAttentionInterface.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_s, this.training ? _attention_dropout : 0f, _softmax_scale, _causal).@out;
                return output.view([batch_size, -1, .. output.shape[1..]]);
            }
            else {
                long nheads = qkv.shape[^2];

                var x = qkv.view(qkv.size(0), qkv.size(1), -1);
                var (x_unpad, indices, cu_seqlens, max_s) = BertPadding.unpad_input(x, key_padding_mask);
                x_unpad = x_unpad.view(x_unpad.size(0), 3, nheads, -1);

                var output_unpad = FlashAttentionInterface.flash_attn_varlen_qkvpacked_func(x_unpad, cu_seqlens, max_s, this.training ? _attention_dropout : 0f, _softmax_scale, _causal).@out;
                output_unpad = output_unpad.view(output_unpad.size(0), -1);

                var output = BertPadding.pad_input(output_unpad, indices, (int)batch_size, (int)seqlen);
                return output.view(output.size(0), output.size(1), nheads, -1);
            }
        }
    }
}