import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from evodiff.utils import Tokenizer
from sequence_models.constants import MSA_AAS
import torch.nn as nn
import torch.nn.functional as F


class OAMaskedCrossEntropyLoss(CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.
    Evaluates the cross-entropy loss at specified locations in a sequence
    When reweight = True, reweights CE according to Hoogeboom et al.;
    reweight term = 1/(D-t+1)
    Shape:
        Inputs:
            - pred: (N, L, n_tokens) # N: batch size
            - tgt: (N, L)
            - mask: (N, L) boolean
            - timestep (N, L) output from OAMaskCollater
            - input mask (N, L)
            - weight: (C, ): class weights for nn.CrossEntropyLoss

    Returns
        ce_losses
        nll_losses
    """
    def __init__(self, weight=None, reduction='none', reweight=True, tokenizer=Tokenizer()):
        self.reweight=reweight
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, tgt, mask, timesteps, input_mask):
        # Make sure we have that empty last dimension
        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
            input_mask = input_mask.unsqueeze(-1)
        # Make sure mask is boolean
        mask = mask.bool()
        input_mask = input_mask.bool() # padded seq
        # Select
        mask_tokens = mask.sum() # masked tokens
        nonpad_tokens = input_mask.sum(dim=1) # nonpad tokens
        p = torch.masked_select(pred, mask).view(mask_tokens, -1) # [T x K] predictions for each mask char
        t = torch.masked_select(tgt, mask.squeeze()) # [ T ] true mask char
        loss = super().forward(p, t) # [ T ] loss per mask char
        # Calculate reweighted CE loss and NLL loss
        nll_losses = loss.sum()
        if self.reweight: # Uses Hoogeboom OARDM reweighting term
            rwt_term = 1. / timesteps
            rwt_term = rwt_term.repeat_interleave(timesteps)
            _n_tokens = nonpad_tokens.repeat_interleave(timesteps)
            ce_loss = _n_tokens * rwt_term * loss
            ce_losses = ce_loss.sum()  # reduce mean
        else:
            ce_losses = nll_losses
        return ce_losses, nll_losses.to(torch.float64)



class RewardLoss(nn.Module):
    def __init__(self, reward_weight=0.1, tokenizer=Tokenizer()):
        super().__init__()
        self.loss_func = OAMaskedCrossEntropyLoss()
        self.tokenizer = tokenizer
        self.reward_weight = reward_weight

    def forward(self, outputs, targets, mask, timestep, input_mask):
        base_loss = self.loss_func(outputs, targets, mask, timestep, input_mask)
        
        # calculate protion of target amino acids
        # 谷氨酸、天冬氨酸、賴氨酸、精氨酸、絲氨酸、蘇氨酸、天門冬醯胺、麩醯胺酸、雙硫鍵
        amino_acids = ['E', 'D', 'K', 'R', 'S', 'T', 'N', 'Q', 'C']  
        aa_indices = [self.tokenizer.a_to_i[aa] for aa in amino_acids]
        
        probs = F.softmax(outputs, dim=-1)
        reward_probs = probs[:, :, aa_indices].sum(dim=-1)
        reward_loss = -torch.log(reward_probs).mean() # 直接算太大了

        return base_loss + self.reward_weight * reward_loss


