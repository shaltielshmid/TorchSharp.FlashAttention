using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.FlashAttention.BertPaddingFunctions {

    internal class IndexFirstAxis : torch.autograd.SingleTensorFunction<IndexFirstAxis> {
        public override string Name => nameof(IndexFirstAxis);

        public override List<torch.Tensor?> backward(torch.autograd.AutogradContext ctx, torch.Tensor grad_output) {
            var indices = ctx.get_saved_variables()[0];
            long firstAxisDim = (long)ctx.get_data("first_axis_dim");

            var otherShape = grad_output.shape[1..];

            // rearrange(grad_output, "b ... -> b (...)")
            grad_output = grad_output.view(grad_output.size(0), -1);
            var grad_input = torch.zeros(new[] { firstAxisDim, grad_output.shape[1] }, dtype: grad_output.dtype, device: grad_output.device);

            // repeat(indices, "z -> z d", d=grad_output.shape[1])
            var repeatedIndices = indices.unsqueeze(1).expand(-1, grad_output.shape[1]);

            grad_input.scatter_(0, repeatedIndices, grad_output);
            return new() { grad_input.reshape(otherShape.Prepend(firstAxisDim).ToArray()), null };
        }

        public static torch.Tensor apply(torch.Tensor input, torch.Tensor indices) {
            return torch.autograd.SingleTensorFunction<IndexFirstAxis>.apply(input, indices);
        }

        public override torch.Tensor forward(torch.autograd.AutogradContext ctx, params object[] vars) {
            var input = (torch.Tensor)vars[0];
            var indices = (torch.Tensor)vars[1];

            ctx.save_for_backward(new() { indices });
            ctx.save_data("first_axis_dim", input.shape[0]);

            var otherShape = input.shape[1..];
            long secondDim = new torch.Size(otherShape).numel();

            // rearrange "b ... -> b (...)"
            var rearrangedInput = input.view(input.size(0), -1);
            // repeat "z -> z d", d = secondDim
            var repeatedIndices = indices.unsqueeze(1).expand(-1, secondDim);

            return torch.gather(rearrangedInput, 0, repeatedIndices).view(otherShape.Prepend(-1).ToArray());
        }
    }
}