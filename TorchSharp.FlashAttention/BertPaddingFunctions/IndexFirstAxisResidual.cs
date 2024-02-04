using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.FlashAttention.BertPaddingFunctions {

    internal class IndexFirstAxisResidual : torch.autograd.MultiTensorFunction<IndexFirstAxisResidual> {
        public override string Name => nameof(IndexFirstAxisResidual);

        public override List<torch.Tensor> backward(torch.autograd.AutogradContext ctx, List<torch.Tensor> grad_outputs) {
            var indices = ctx.get_saved_variables()[0];
            long firstAxisDim = (long)ctx.get_data("first_axis_dim");

            var grad_output = grad_outputs[0];
            var grad_input = grad_outputs[1];

            indices = indices.reshape([indices.shape[0], .. Enumerable.Repeat(1, (int)grad_output.ndim - 1)]);
            indices = indices.expand_as(grad_output);

            grad_input.scatter_add_(0, indices, grad_output);
            return [grad_input.reshape([firstAxisDim, .. grad_output.shape[1..]]), null];
        }

        public static List<torch.Tensor> apply(torch.Tensor input, torch.Tensor indices) {
            return torch.autograd.MultiTensorFunction<IndexFirstAxisResidual>.apply(input, indices);
        }

        public override List<torch.Tensor> forward(torch.autograd.AutogradContext ctx, params object[] vars) {
            var input = (torch.Tensor)vars[0];
            var indices = (torch.Tensor)vars[1];

            ctx.save_for_backward([indices]);
            ctx.save_data("first_axis_dim", input.shape[0]);

            var output = input[indices];
            return [output, input.detach()];
        }
    }
}