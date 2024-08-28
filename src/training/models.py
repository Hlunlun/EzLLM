import torch.nn as nn
import torch
from evodiff.utils import Tokenizer





class EnhancedModel(nn.Module):
    def __init__(self, base_model, tokenizer = Tokenizer()):
        super().__init__()
        self.base_model = base_model
        self.interaction_bias = nn.Parameter(torch.zeros(len(tokenizer)))
        self.target_aa = ['E', 'D', 'K', 'R', 'S', 'T', 'N', 'Q']
        self.target_indices = [tokenizer.token_to_id[aa] for aa in self.target_aa]
        self.interaction_bias.data[self.target_indices] = 0.1  # 初始偏好

    def forward(self, *args, **kwargs):
        outputs = self.base_model(*args, **kwargs)
        return outputs + self.interaction_bias.unsqueeze(0).unsqueeze(0)









