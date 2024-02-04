using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.FlashAttention.BertPaddingFunctions;

namespace TorchSharp.FlashAttention {
    public static class BertPadding {

        /// <summary>
        /// Alias of `BertPadding.UnpadInput` returning a desconstructable tuple instead of a record.
        /// </summary>
        /// <param name="hidden_states">(batch, seqlen, ...)</param>
        /// <param name="attention_mask">(batch, seqlen), bool / int, 1 means valid and 0 means not valid.</param>
        /// <returns>
        /// hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        /// indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        /// cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        /// max_seqlen_in_batch: int
        /// </returns>
        public static (torch.Tensor hidden_states, torch.Tensor indices, torch.Tensor cu_seqlens, int max_seqlen_in_batch) unpad_input(torch.Tensor hidden_states, torch.Tensor attention_mask) {
            var res = UnpadInput(hidden_states, attention_mask);
            return (res.HiddenStates, res.Indices, res.CU_SeqLens, res.MaxSeqLenInBatch);
        }

        /// <summary>
        /// </summary>
        /// <param name="hidden_states">(batch, seqlen, ...)</param>
        /// <param name="attention_mask">(batch, seqlen), bool / int, 1 means valid and 0 means not valid.</param>
        /// <returns>
        /// hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        /// indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        /// cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        /// max_seqlen_in_batch: int
        /// </returns>
        public static UnpaddedInput UnpadInput(torch.Tensor hidden_states, torch.Tensor attention_mask) {
            var seqlensInBatch = attention_mask.sum(-1, type: torch.int32);
            var indices = torch.nonzero(attention_mask.flatten()).flatten();
            int maxSeqlensInBatch = seqlensInBatch.max().item<int>();
            var cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlensInBatch, 0, torch.int32), new long[] { 1, 0 });

            // Comment explaining why we use custom forward for indexing copied from python code:
            // TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
            // bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
            // times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
            // index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
            // so we write custom forward and backward to make it a bit faster.
            return new(
                IndexFirstAxis.apply(hidden_states.view(new[] { -1L }.Concat(hidden_states.shape[2..]).ToArray()), indices),
                indices,
                cu_seqlens,
                maxSeqlensInBatch
            );
        }

        /// <summary>
        /// Alias of `BertPadding.UnpadInputForConcatenatedSequences` returning a desconstructable tuple instead of a record.
        /// </summary>
        /// <param name="hidden_states">(batch, seqlen, ...)</param>
        /// <param name="attention_mask">(batch, seqlen), bool / int, 1 means valid and 0 means not valid.</param>
        /// <returns>
        /// hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        /// indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        /// cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        /// max_seqlen_in_batch: int
        /// </returns>
        public static (torch.Tensor hidden_states, torch.Tensor indices, torch.Tensor cu_seqlens, int max_seqlen_in_batch) unpad_input_for_concatenated_sequences(torch.Tensor hidden_states, torch.Tensor attention_mask_in_length) {
            var res = UnpadInput(hidden_states, attention_mask_in_length);
            return (res.HiddenStates, res.Indices, res.CU_SeqLens, res.MaxSeqLenInBatch);
        }

        /// <summary>
        /// Supports concatenating short samples in one sequence. The attention_mask_in_length is utilized to mask other short samples. It helps efficient training of variant lengths-based samples (e.g., the supervised fine-tuning task in large language model).
        /// The motivation for this function is explained[here] (https://github.com/Dao-AILab/flash-attention/issues/432#issuecomment-1668822286).
        /// For example, if batch = 3 and seqlen = 6, the attention_mask_in_length is:
        /// ```
        ///        [
        ///          [2, 3, 0, 0, 0, 0],
        ///          [3, 2, 0, 0, 0, 0],
        ///          [6, 0, 0, 0, 0, 0]
        ///        ]
        ///        ```
        ///    , which refers to the 3D-attention mask:
        ///        ```
        ///        [
        ///          [
        ///            [1, 0, 0, 0, 0, 0],
        ///            [1, 1, 0, 0, 0, 0],
        ///            [0, 0, 1, 0, 0, 0],
        ///            [0, 0, 1, 1, 0, 0],
        ///            [0, 0, 1, 1, 1, 0],
        ///            [0, 0, 0, 0, 0, 1]
        ///          ],
        ///          [
        ///            [1, 0, 0, 0, 0, 0],
        ///            [1, 1, 0, 0, 0, 0],
        ///            [1, 1, 1, 0, 0, 0],
        ///            [0, 0, 0, 1, 0, 0],
        ///            [0, 0, 0, 1, 1, 0],
        ///            [0, 0, 0, 0, 0, 1]
        ///          ],
        ///          [
        ///            [1, 0, 0, 0, 0, 0],
        ///            [1, 1, 0, 0, 0, 0],
        ///            [1, 1, 1, 0, 0, 0],
        ///            [1, 1, 1, 1, 0, 0],
        ///            [1, 1, 1, 1, 1, 0],
        ///            [1, 1, 1, 1, 1, 1]
        ///          ]
        ///        ]
        ///        ```.
        /// </summary>
        /// <param name="hidden_states">(batch, seqlen, ...)</param>
        /// <param name="attention_mask_in_length">(batch, seqlen), int, a nonzero number (e.g., 1, 2, 3, etc.) means length of concatenated sequence in b-th batch, and 0 means none.</param>
        /// <returns>
        /// hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        /// indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        /// cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        /// max_seqlen_in_batch: int
        /// </returns>
        public static UnpaddedInput UnpadInputForConcatenatedSequences(torch.Tensor hidden_states, torch.Tensor attention_mask_in_length) {
            var length = attention_mask_in_length.sum(dim: -1);
            var seqlen = attention_mask_in_length.size(-1);
            var attentionMask2D = torch.arange(seqlen, device: length.device, dtype: length.dtype).expand(length.size(0), seqlen) < length.unsqueeze(1);
            var realIndicesIdx = torch.nonzero(attention_mask_in_length.flatten()).flatten();
            var seqlensInBatch = attention_mask_in_length.flatten()[realIndicesIdx];
            var indices = torch.nonzero(attentionMask2D.flatten()).flatten();
            int maxSeqlensInBatch = seqlensInBatch.max().item<int>();
            var cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlensInBatch, 0, torch.int32), new[] { 1L, 0L });

            // Comment explaining why we use custom forward for indexing copied from python code:
            // TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
            // bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
            // times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
            // index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
            // so we write custom forward and backward to make it a bit faster.
            return new(
                IndexFirstAxis.apply(hidden_states.view(new[] { -1L }.Concat(hidden_states.shape[2..]).ToArray()), indices),
                indices,
                cu_seqlens,
                maxSeqlensInBatch
            );
        }

        /// <summary>
        /// Alias of `BertPadding.PadInput`
        /// </summary>
        /// <param name="hidden_states">(total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.</param>
        /// <param name="indices">(total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.</param>
        /// <param name="batch">int, batch size for the padded sequence.</param>
        /// <param name="seqlen">int, maximum sequence length for the padded sequence.</param>
        /// <returns>hidden_states: (batch, seqlen, ...)</returns>
        public static torch.Tensor pad_input(torch.Tensor hidden_states, torch.Tensor indices, int batch, int seqlen) {
            return PadInput(hidden_states, indices, batch, seqlen);
        }

        /// <summary>
        /// </summary>
        /// <param name="hidden_states">(total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.</param>
        /// <param name="indices">(total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.</param>
        /// <param name="batch">int, batch size for the padded sequence.</param>
        /// <param name="seqlen">int, maximum sequence length for the padded sequence.</param>
        /// <returns>hidden_states: (batch, seqlen, ...)</returns>
        public static torch.Tensor PadInput(torch.Tensor hidden_states, torch.Tensor indices, int batch, int seqlen) {
            var output = IndexPutFirstAxis.apply(hidden_states, indices, batch * seqlen);
            return output.view(new[] { batch, -1L }.Concat(output.shape[1..]).ToArray());
        }
    }

    public record class UnpaddedInput(torch.Tensor HiddenStates, torch.Tensor Indices, torch.Tensor CU_SeqLens, int MaxSeqLenInBatch);
}