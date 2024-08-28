import argparse
import json
import os
from datetime import datetime, timedelta
import pathlib
import numpy as np
# torch
import torch
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data import Subset
# evodiff
from evodiff.collaters import D3PMCollaterMSA
from evodiff.utils import Tokenizer
from evodiff.losses import  D3PMCELoss,  D3PMLVBLossMSA
from evodiff.model import MSATransformerTime
from evodiff.metrics import MaskedAccuracyMSA
# self-defined
from dataset import A3MMSADataset
# sequencc_models
from sequence_models.esm import MSATransformer
from sequence_models.constants import MSA_ALPHABET
from sequence_models.collaters import MSAAbsorbingCollater
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.losses import MaskedCrossEntropyLossMSA
from sequence_models.utils import warmup, transformer_lr




home = str(pathlib.Path.home())

parser = argparse.ArgumentParser()
parser.add_argument('config_fpath')
parser.add_argument('out_fpath', type=str, nargs='?', default=os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/')
# parser.add_argument('out_fpath', type=str, nargs='?', default=home + '/model_output/openfold_checkpoints/')
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
parser.add_argument('-off', '--offset', default=0, type=int,
                    help='Number of GPU devices to skip.')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--tie_weights', action='store_true')
parser.add_argument('--task', default=None)
parser.add_argument('--dataset', default=None)
parser.add_argument('--aml', action='store_true')  # Set true to do multi-node training on amlk8s
parser.add_argument('-sd', '--state_dict', default=None)
parser.add_argument('--decay', action='store_true')
parser.add_argument('--dummy', required=False)
parser.add_argument('--mask', default='blosum')
parser.add_argument('--checkpoint_freq', type=float, default=120)  # in minutes
parser.add_argument('--weight-save-freq', type=float, default=None)  # in minutes
parser.add_argument('--log-freq', type=float, default=1000)  # in steps
parser.add_argument('--reweighting_term', type=float, default=0.001) # lambda from D3PM
parser.add_argument('--selection-type', type=str, default='MaxHamming') # MaxHamming or random

args = parser.parse_args()






''' GPU '''
# Clear the CUDA cache to free up memory
torch.cuda.empty_cache()
# Set max_split_size_mb to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(args.gpus)
device = torch.device(f'cuda:{args.gpus}')


''' Configuration '''
with open(args.config_fpath, 'r') as f:
    config = json.load(f)
selection_type = args.selection_type
d_embed = config['d_embed']
d_hidden = config['d_hidden']
n_layers = config['n_layers']
n_heads = config['n_heads']
bucket_size = config['bucket_size']
max_tokens = config['max_tokens']
max_batch_size = config['max_batch_size']
epochs = config['epochs']
lr = config['lr']
warmup_steps = config['warmup']
max_square_tokens = config['max_square_tokens']
n_sequences = config['n_sequences']
min_depth = config['n_sequences'] # Will filter out sequences smaller than this number
max_seq_len = config['max_seq_len']
config['decay'] = args.decay


''' Tokenizer '''
tokenizer = Tokenizer()
padding_idx = tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
masking_idx = tokenizer.mask_id
gap_idx = tokenizer.gap_id
print(tokenizer.alphabet)
print('Using {} as padding index'.format(padding_idx))
print('Using {} as masking index'.format(masking_idx))
print('Using {} as gap index'.format(gap_idx))



''' Dataset '''
collater = MSAAbsorbingCollater(alphabet=MSA_ALPHABET)
dataset = A3MMSADataset(selection_type, n_sequences, max_seq_len, data_dir=data_dir, min_depth=min_depth)
train_size = len(dataset)
print("TRAIN SIZE:", train_size, rank)
random_ind = np.random.choice(train_size, size=(train_size - 10000), replace=False)
ds_train = Subset(dataset, random_ind)

metadata = np.array(dataset.lengths)
train_idx = ds_train.indices
#print(train_idx)
len_train = metadata[train_idx]

len_train = np.minimum(len_train, max_seq_len)

train_sortish_sampler = SortishSampler(len_train, bucket_size, num_replicas=args.world_size, rank=rank)
train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, len_train,
                                    max_square_tokens=max_square_tokens, msa_depth=n_sequences)
dl_train = DataLoader(dataset=ds_train,
                        batch_sampler=train_sampler,
                        collate_fn=collater,
                              num_workers=8)


val_ind = np.delete(np.arange(train_size), random_ind)
ds_valid = Subset(dataset, val_ind)
valid_idx = ds_valid.indices
len_valid = metadata[valid_idx]
len_valid = np.minimum(len_valid, max_seq_len)

valid_sortish_sampler = SortishSampler(len_valid, bucket_size, num_replicas=1, rank=0)
valid_sampler = ApproxBatchSampler(valid_sortish_sampler, max_tokens, max_batch_size, len_valid,
                                    max_square_tokens=max_square_tokens, msa_depth=n_sequences)

dl_valid = DataLoader(dataset=ds_valid,
                        batch_sampler=valid_sampler,
                        collate_fn=collater,
                        num_workers=8)



''' Model '''
diffusion_timesteps = None # Not input to model
if args.mask == 'oadm':
    model = MSATransformer(d_embed, d_hidden, n_layers, n_heads, use_ckpt=True, n_tokens=len(MSA_ALPHABET),
                            padding_idx=padding_idx, mask_idx=masking_idx).cuda()
else:
    model = MSATransformerTime(d_embed, d_hidden, n_layers, n_heads, timesteps=diffusion_timesteps, use_ckpt=True,
                                n_tokens=len(MSA_ALPHABET), padding_idx=padding_idx, mask_idx=masking_idx).cuda()
    optimizer = Adam(model.parameters(), lr=lr)
if args.decay:
    scheduler = LambdaLR(optimizer, transformer_lr(warmup_steps))
else:
    scheduler = LambdaLR(optimizer, warmup(warmup_steps))
scaler = GradScaler()





