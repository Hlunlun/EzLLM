import torch
import os
from evodiff.pretrained import OA_DM_640M



checkpoint = OA_DM_640M()
model, collater, tokenizer, scheme = checkpoint


model.load_state_dict(torch.load('models/epoch1_lr0.0001_accum8_warmup20_weight-decay1e-05/model_epoch0_accu36.5882.pth'))


from evodiff.generate import generate_oaardm

seq_len = 100
tokeinzed_sample, generated_sequence = generate_oaardm(model, tokenizer, seq_len, batch_size=1, device='cpu')
print("Generated sequence:", generated_sequence)


