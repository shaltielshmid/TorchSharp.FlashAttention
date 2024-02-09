using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.FlashAttention.BertPaddingFunctions {

    internal class IndexPutFirstAxis : torch.autograd.SingleTensorFunction<IndexPutFirstAxis> {
        public override string Name => nameof(IndexPutFirstAxis);

        public override List<torch.Tensor> backward(torch.autograd.AutogradContext ctx, torch.Tensor grad_output) {
            var indices = ctx.get_saved_variables()[0];

            return new() { grad_output[indices], null, null };
        }

        public static torch.Tensor apply(torch.Tensor values, torch.Tensor indices, long first_axis_dim) {
            return torch.autograd.SingleTensorFunction<IndexPutFirstAxis>.apply(values, indices, first_axis_dim); 
        }

        public override torch.Tensor forward(torch.autograd.AutogradContext ctx, params object[] vars) {
            var values = (torch.Tensor)vars[0];
            var indices = (torch.Tensor)vars[1];
            long firstAxisDim = (long)vars[2];

            ctx.save_for_backward(new() { indices });

            var output = torch.zeros(
                new[] { firstAxisDim }.Concat(values.shape[1..]).ToArray(),
                dtype: values.dtype,
                device: values.device
            );

            output[indices] = values;
            return output;
        }
    }
}