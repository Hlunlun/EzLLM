import numpy as np
import torch
from evodiff.utils import Tokenizer
from sequence_models.constants import PAD, PROTEIN_ALPHABET, GAP
from typing import List, Any, Iterable
from sequence_models.constants import PAD, GAP, START, STOP, MASK, MSA_PAD, PROTEIN_ALPHABET



def _pad(tokenized, value, dim=2):
    """
    Utility function that pads batches to the same length.

    tokenized: list of tokenized sequences
    value: pad index
    """
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    if dim == 3: # dim = 3 (one hot)
        categories = tokenized[0].shape[-1]
        output = torch.zeros((batch_size, max_len, categories)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t), :] = t
    elif dim == 2: # dim = 2 (tokenized)
        output = torch.zeros((batch_size, max_len)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t)] = t
    else:
        print("padding not supported for dim > 3")
    return output

# def _pad_msa(tokenized, num_seq, max_len, value, dim=3):
#     """Utility function that pads batches to the same length."""
#     batch_size = len(tokenized)
#     if dim == 4: # one hot MSA
#         categories = tokenized[0].shape[-1] # last dim is one hot
#         output = torch.zeros((batch_size, num_seq, max_len, categories), dtype=torch.long) + value
#         for i in range(batch_size):
#             output[i, :, :len(tokenized[i][0]), :] = tokenized[i]
#     elif dim == 3: # tokenized MSA
#         output = torch.zeros((batch_size, num_seq, max_len), dtype=torch.long) + value
#         for i in range(batch_size):
#             output[i, :, :len(tokenized[i][0])] = tokenized[i]
#     else:
#         print("padding not supported for dim > 4")
#     return output




class OAMaskCollater(object):
    """
    OrderAgnosic Mask Collater for masking batch data according to Hoogeboom et al. OA ARDMS
    inputs:
        sequences : list of sequences
        inputs_padded: if inputs are padded (due to truncation in Simple_Collater) set True (default False)

    OA-ARM variables:
        D : possible permutations from 0.. max length
        t : randomly selected timestep

    outputs:
        src : source  masked sequences (model input)
        timesteps: (D-t+1) term
        tokenized: tokenized sequences (target seq)
        masks: masks used to generate src
    """
    def __init__(self, tokenizer=Tokenizer()):
        self.tokenizer = tokenizer


    def __call__(self, sequences):
        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        max_len = max(len(t) for t in tokenized)
        src=[]
        timesteps = []
        masks=[]
        mask_id = torch.tensor(self.tokenizer.mask_id, dtype=torch.int64)
        for i,x in enumerate(tokenized):
            # Randomly generate timestep and indices to mask
            D = len(x) # D should have the same dimensions as each sequence length
            if D <= 1:  # for sequence length = 1 in dataset
                t = 1
            else:
                t = np.random.randint(1, D) # randomly sample timestep
            num_mask = (D-t+1) # from OA-ARMS
            # Append timestep
            timesteps.append(num_mask)
            # Generate mask
            mask_arr = np.random.choice(D, num_mask, replace=False) # Generates array of len num_mask
            index_arr = np.arange(0, max_len) #index array [1...seq_len]
            mask = np.isin(index_arr, mask_arr, invert=False).reshape(index_arr.shape) # mask bools indices specified by mask_arr
            # Mask inputs
            mask = torch.tensor(mask, dtype=torch.bool)
            masks.append(mask)
            x_t = ~mask[0:D] * x + mask[0:D] * mask_id
            src.append(x_t)
        # PAD out
        src = _pad(src, self.tokenizer.pad_id)
        masks = _pad(masks*1,0) #, self.seq_length, 0)
        tokenized = _pad(tokenized, self.tokenizer.pad_id)
        return (src.to(torch.long), torch.tensor(timesteps), tokenized.to(torch.long), masks)




class EnhancedOAMaskCollater(OAMaskCollater):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, tokenizer=Tokenizer())
        self.target_aa = ['E', 'D', 'K', 'R', 'S', 'T', 'N', 'Q']
        self.target_indices = [tokenizer.token_to_id[aa] for aa in self.target_aa]

    def __call__(self, batch):
        src, timestep, tgt, mask = super().__call__(batch)
        
        # 增加保留目标氨基酸的概率
        target_mask = torch.isin(src, torch.tensor(self.target_indices).to(src.device))
        mask = mask & (~target_mask | (torch.rand_like(mask.float()) > 0.7)).bool()

        return src, timestep, tgt, mask



def _pad_msa(tokenized: List, num_seq: int, max_len: int, value: int) -> torch.Tensor:
    """Utility function that pads batches to the same length."""
    batch_size = len(tokenized)
    num_seq = max([len(m) for m in tokenized])
    output = torch.zeros((batch_size, num_seq, max_len), dtype=torch.long) + value
    for i in range(batch_size):
        tokenized[i] = torch.LongTensor(np.array(tokenized[i]))
        output[i, :len(tokenized[i]), :len(tokenized[i][0])] = tokenized[i]
    return output



class MSAAbsorbingCollater(object):
    """Collater for MSA Absorbing Diffusion model.
    Based on implementation described by Hoogeboom et al. in "Autoregressive Diffusion Models"
    https://doi.org/10.48550/arXiv.2110.02037

    Parameters:
        alphabet: str,
            protein alphabet to use
        pad_token: str,
            pad_token to use to pad MSAs, default is PAD token from sequence_models.constants
        num_seqs: int,
            number of sequences to include in each MSA

    Input (list): a batch of Multiple Sequence Alignments (MSAs), each MSA contains 64 sequences
    Output:
        src (torch.LongTensor): corrupted input + padding
        tgt (torch.LongTensor): input + padding
        mask (torch.LongTensor): 1 where tgt is not padding
    """

    def __init__(self, alphabet: str, pad_token=MSA_PAD, num_seqs=64, bert=False):
        self.tokenizer = Tokenizer(alphabet)
        self.pad_idx = self.tokenizer.alphabet.index(pad_token)
        self.num_seqs = num_seqs
        self.bert = bert
        if bert:
            self.choices = [self.tokenizer.alphabet.index(a) for a in PROTEIN_ALPHABET + GAP]

    def __call__(self, batch_msa):
        tgt = list(batch_msa)
        src = tgt.copy()

        longest_msa = 0
        batch_size = len(batch_msa)
        mask_ix = []
        mask_iy = []
        for i in range(batch_size):
            # Tokenize MSA
            tgt[i] = [self.tokenizer.tokenize(s) for s in tgt[i]]
            src[i] = [self.tokenizer.tokenize(s) for s in src[i]]

            curr_msa = src[i]

            curr_msa = np.asarray(curr_msa)
            length, depth = curr_msa.shape  # length = number of seqs in MSA, depth = # AA in MSA

            curr_msa = curr_msa.flatten()  # Flatten MSA to 1D to mask tokens
            d = len(curr_msa)  # number of residues in MSA
            if not self.bert:
                t = np.random.choice(d)  # Pick timestep t
                t += 1  # ensure t cannot be 0
                num_masked_tokens = d - t + 1
                mask_idx = np.random.choice(d, num_masked_tokens, replace=False)
            else:
                num_corr_tokens = int(np.round(0.15 * d))
                corr_idx = np.random.choice(d, num_corr_tokens, replace=False)
                num_masked_tokens = int(np.round(0.8 * num_corr_tokens))
                num_mut_tokens = int(np.round(0.1 * num_corr_tokens))
                mask_idx = corr_idx[:num_masked_tokens]
                muta_idx = corr_idx[-num_mut_tokens:]
                for idx in muta_idx:
                    choices = list(set(self.choices) - set(curr_msa[[idx]]))
                    curr_msa[idx] = np.random.choice(choices)
                mask_ix.append(corr_idx // depth)
                mask_iy.append(corr_idx % depth)

            curr_msa[mask_idx] = self.tokenizer.mask_id
            curr_msa = curr_msa.reshape(length, depth)

            src[i] = list(curr_msa)

            longest_msa = max(depth, longest_msa)  # Keep track of the longest MSA for padding

        # Pad sequences
        src = _pad_msa(src, self.num_seqs, longest_msa, self.pad_idx)
        tgt = _pad_msa(tgt, self.num_seqs, longest_msa, self.pad_idx)
        if self.bert:
            mask = torch.zeros_like(src)
            for i in range(len(mask_ix)):
                mask[i, mask_ix[i], mask_iy[i]] = 1
            mask = mask.bool()
        else:
            mask = (src == self.tokenizer.mask_id)

        return src, tgt, mask



