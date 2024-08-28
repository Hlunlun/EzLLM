import os
import re
import json
import torch
from evodiff.pretrained import OA_DM_640M
from collaters import OAMaskCollater
from dataset import FASTADataset
from evodiff.utils import Tokenizer
from torch.utils.data import DataLoader
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW
from losses import OAMaskedCrossEntropyLoss
from sequence_models.metrics import MaskedAccuracy # 不知道怎麼用
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from sequence_models.utils import warmup 
from torch.nn import CrossEntropyLoss
import argparse
import torch.nn.functional as F
import pandas as pd   
import random
import numpy as np
import heapq
from evodiff.utils import Tokenizer
# from torch.utils.tensorboard import SummaryWriter



from evodiff.utils import Tokenizer, run_omegafold, clean_pdb, run_tmscore #, wrap_dr_bert, read_dr_bert_output


''' Parameter '''
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=int, default=2)
parser.add_argument('--range', type=str, default='species')
parser.add_argument('--random-seed', type=int, default=42)
parser.add_argument('--lr', type=int, default=1e-4)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--train-batch-size', type=int, default=1)
parser.add_argument('--validation-batch-size', type=int, default=1)
parser.add_argument('--test-batch-size', type=int, default=1)
parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
parser.add_argument('--warmup-steps', type=int, default=20)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--save-steps', type=int, default=10)
parser.add_argument('--reweight', type=str, choices=['True', 'False'], default='False')
parser.add_argument('--model-type', type=str, default='oa_dm_640M',
                        help='Choice of: carp_38M carp_640M esm1b_650M \
                              oa_dm_38M oa_dm_640M   lr_ar_640M')
args = parser.parse_args()
print(args)


''' GPU '''
# Clear the CUDA cache to free up memory
torch.cuda.empty_cache()
# Set max_split_size_mb to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(args.gpus)
device = torch.device(f'cuda:{args.gpus}')


''' Model '''
checkpoint = OA_DM_640M()
model, collater, tokenizer, scheme = checkpoint
model.eval().cuda()
model.to(device) # all tensor on same device 20240718 --LUN
tokenizer = Tokenizer()
padding_idx = tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
masking_idx = tokenizer.mask_id
optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
scaler = GradScaler()
scheduler = LambdaLR(optimizer, warmup(args.warmup_steps), verbose=False)

'''
# Warmup function
def warmup(warmup_steps):
    return lambda epoch: float(epoch) / float(max(1, warmup_steps)) if epoch < warmup_steps else 1.0
# Learning rate decay function
def lr_decay(epoch):
    return 0.95 ** epoch
# Initialize the optimizer
# optimizer = AdamW(model.parameters(), lr=args.lr)
# Create the learning rate scheduler
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_decay(epoch) * warmup(args.warmup_steps)(epoch), verbose=False)
'''





''' Dataset '''
data_dir = f'/home/chialun/projects/evodiff/data/ec_{args.range}'
collater = OAMaskCollater()
dataset = FASTADataset(dataset_path=data_dir)
train_dataset, untrain_dataset = train_test_split(dataset, test_size=0.2, random_state=args.random_seed)
test_dataset, validation_dataset = train_test_split(untrain_dataset, test_size=0.5, random_state=args.random_seed)
train_loader = DataLoader(train_dataset, collate_fn=collater, batch_size=args.train_batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, collate_fn=collater, batch_size=args.validation_batch_size, shuffle=False)

# count the number that amino acid be masked
# src, timesteps, tokenized, masks = collater.__call__(dataset)
# print(src.shape)
# print((src[0] == 28).sum().item())



'''' Loss Function '''
args.reweight = args.reweight == 'True'
loss_func = OAMaskedCrossEntropyLoss(reweight=args.reweight)
accu_func = MaskedAccuracy()
accus = []
losses = []
accus_eval = []
losses_eval = []



''' TensorBoard '''
# writer = SummaryWriter(f'runs/epoch{args.epochs}_lr{args.lr}_accum{args.gradient_accumulation_steps}_warmup{args.warmup_steps}_weight-decay{args.weight_decay}')




def calculate_reward(preds):
    tokenizer = Tokenizer()
    # 氫鍵和鹽橋的形成：具有較高的氫鍵形成能力（如絲氨酸、蘇氨酸）和能形成穩定鹽橋（如帶正電的賴氨酸、精氨酸與帶負電的谷氨酸、天冬氨酸）可以提高蛋白質的熱穩定性。
    # 谷氨酸、天冬氨酸、賴氨酸、精氨酸、絲氨酸、蘇氨酸、天門冬醯胺、麩醯胺酸、半胱氨酸
    # 雙硫鍵: 半胱氨酸（Cys）是形成二硫鍵的氨基酸 
    target_aa = ['E', 'D', 'K', 'R', 'S', 'T', 'N', 'Q', 'C'] 
    target_indices = torch.tensor([tokenizer.a_to_i[aa] for aa in target_aa]).to(preds.device)
    return (preds.unsqueeze(-1) == target_indices).any(dim=-1).float().mean()



''' Save Path '''
model_dir = f'models/{args.model_type}'
os.makedirs(model_dir, exist_ok = True)
top_models = []




''' To save in best state dict '''
config_file_path = os.path.join(model_dir, f'config_{args.model_type}.json')
if os.path.exists(config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    model_pre = [pth for pth in os.listdir(model_dir) if pth.endswith('.pth')][0]
    best_accuracy = og_accuracy = float(re.search(r'accu([\d\.]+).pth', model_pre).group(1))
else:
    best_accuracy = og_accuracy = 0.0
best_model_state = None
best_optimizer_state = None
best_scheduler_state = None
best_scaler_state = None
best_epoch = 0



''' Training Loop '''
for ep in range(args.epochs):
    model.train()
    pbar = tqdm(train_loader)
    pbar.set_description(f"Training epoch [{ep+1}/{args.epochs}]")
    
    accu_mean = 0.0 
    loss_mean = 0.
    total_steps = 0  
    
    for src, timestep, tgt, mask in pbar:

        timestep = timestep.to(device)
        src = src.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)
        n_tokens = mask.sum()
        input_mask = (src != padding_idx).float()

        outputs = model(src, timestep, input_mask=input_mask.unsqueeze(-1))

        ce_loss, nll_loss = loss_func(outputs, tgt, mask, timestep, input_mask)  # sum(loss per token)
        loss = ce_loss

        # # Step 1: 计算交叉熵损失
        # loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), tgt.view(-1), reduction='none')
        # # Step 2: 将 loss 恢复为原始形状 (batch_size, seq_len)
        # loss = loss.view(outputs.size(0), outputs.size(1))
        # # Step 3: 仅选择未被 padding 且被 mask 的位置的损失
        # effective_loss = loss * mask * input_mask
        # # Step 4: 求和并平均得到最终的 loss
        # final_loss = effective_loss.sum() / timestep
        # loss = final_loss 

        loss_mean += loss.item()

        preds = torch.argmax(outputs, dim=-1)
        correct = (preds == tgt) * mask  # onlu calculate accuracy where are masked
        accu = correct.sum().float() / mask.sum().float()     
        accu_mean += accu.item() 
        total_steps += 1  # iteration        

        # '''
        reward = calculate_reward(preds)
        # Reinforcement Learning to modify loss
        loss = loss * (1 - reward * 0.1)  # 10% reward weight
        scaler.scale(loss).backward()
        # '''

        # loss = loss / args.gradient_accumulation_steps
        # scaler.scale(loss).backward()        
        # if (total_steps + 1) % args.gradient_accumulation_steps == 0:
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad()
        #     scale = scaler.get_scale()
        #     skip_scheduler = (scale > scaler.get_scale())
        #     if not skip_scheduler:
        #         scheduler.step()

        ''' high precision '''
        # scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_scheduler = (scale > scaler.get_scale())
        if not skip_scheduler:
            scheduler.step()

        pbar.set_postfix(loss =loss_mean / total_steps, accu=accu_mean / total_steps)
    
    accus.append(accu_mean / total_steps)
    losses.append(loss_mean / total_steps)
    
    
    model.eval()
    pbar = tqdm(validation_loader)
    pbar.set_description("Evaluating")
    total_steps = 0
    accu_mean_eval = 0.
    loss_mean_eval = 0.
    for src, timestep, tgt, mask in pbar:
        timestep = timestep.to(device)
        src = src.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)
        n_tokens = mask.sum()
        input_mask = (src != padding_idx).float()

        outputs = model(src, timestep, input_mask=input_mask.unsqueeze(-1))

        ce_loss, nll_loss = loss_func(outputs, tgt, mask, timestep, input_mask)  # sum(loss per token)
        loss = ce_loss

        preds = torch.argmax(outputs, dim=-1)
        correct = (preds == tgt) * mask  # 只保留非掩码位置的准确性
        accu = correct.sum().float() / mask.sum().float()       


        total_steps += 1
        accu_mean_eval += accu.item()
        loss_mean_eval += loss.item()
        pbar.set_postfix(loss = loss_mean_eval / total_steps, accu=accu_mean_eval / total_steps)
    accus_eval.append(accu_mean_eval / total_steps)
    losses_eval.append(loss_mean_eval / total_steps)

    if accu_mean_eval > best_accuracy:
        best_accuracy = accu_mean_eval
        best_model_state = model.state_dict()
        best_optimizer_state = optimizer.state_dict()
        best_scheduler_state = scheduler.state_dict()
        best_scaler_state = scaler.state_dict() if scaler is not None else None
        best_epoch = ep

    # # Push the new model into the heap
    # heapq.heappush(top_models, (accu_mean_eval, ep, model.state_dict()))
    # # If the heap exceeds 5 models, remove the one with the lowest accuracy
    # if len(top_models) > 5:
    #     heapq.heappop(top_models)


# for accu, ep, model_state in top_models:
#     torch.save(model_state, os.path.join(model_dir, f'model_epoch{ep}_accu{accu:.4f}.pth'))
''' Save model state dict if new model perform better than previous one'''
if best_accuracy > og_accuracy:

    torch.save({
        'model_state_dict': best_model_state,
        'optimizer_state_dict': best_optimizer_state,
        'scheduler_state_dict': best_scheduler_state,
        'scaler_state_dict': best_scaler_state,
    }, os.path.join(model_dir, f'OADM640M_accu{best_accuracy:.4f}.pth'))

    # Collect configuration parameters
    config = {
        'gpus': args.gpus,
        'range': args.range,
        'random_seed': args.random_seed,
        'lr': args.lr,
        'epochs': args.epochs,
        'train_batch_size': args.train_batch_size,
        'validation_batch_size': args.validation_batch_size,
        'test_batch_size': args.test_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_steps': args.warmup_steps,
        'weight_decay': args.weight_decay,
        'save_steps': args.save_steps,
        'reweight': args.reweight,
        'best_step': best_epoch,
    }

    # Save configuration to a JSON file
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)




fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot accuracy data on the left y-axis
ax1.plot(accus, label='Training Accuracy', color = 'deepskyblue')
ax1.plot(accus_eval, label='Evaluating Accuracy', color = 'lightseagreen')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.tick_params(axis='y')

x_ticks = np.arange(1, args.epochs+1, step=4)
ax1.set_xticks(x_ticks) 
ax1.set_xticklabels(x_ticks)

# Create a second y-axis for loss data
ax2 = ax1.twinx()
ax2.plot(losses, label='Training Loss', color='orchid')
ax2.plot(losses_eval, label='Evaluating Loss', color='salmon')
ax2.set_ylabel('Loss')
ax2.tick_params(axis='y')

# Set the title
plt.title(f'Accuracy and Loss - epoch{args.epochs}_lr{args.lr}_accum{args.gradient_accumulation_steps}_warmup{args.warmup_steps}_reweight{args.reweight}_weight-decay{args.weight_decay}')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_transform=fig.transFigure, ncol=4)

plt.grid(True)
plt.savefig(f'performance/accuracy_loss_epoch{args.epochs}_lr{args.lr}_accum{args.gradient_accumulation_steps}_warmup{args.warmup_steps}_reweight{args.reweight}_w-decay{args.weight_decay}.png', dpi=300, bbox_inches='tight')
plt.show()







''' Testing '''

# motif_dict = pd.read_csv(os.path.join(data_dir, 'motif_dict.csv'))
# batch_size = 1

# for seq in test_dataset:

#     ''' search for motif position '''
#     pdb_id, motif_start, motif_end = motif_dict[motif_dict['Sequence'] == seq[0]][['Entry', 'Motif_start', 'Motif_end']].values[0]
#     motif_start, motif_end = int(motif_start), int(motif_end)
#     motif_seq = seq[motif_start:motif_end]
#     motif_tokenized = tokenizer.tokenize((motif_seq,))
    
#     scaffold_length = random.randint(100, 1024)
#     seq_len = scaffold_length + len(motif_seq)

#     new_start = np.random.choice(scaffold_length)
#     sample = torch.zeros((batch_size, seq_len)) + masking_idx # start from all mask
#     new_start = np.random.choice(scaffold_length) # randomly place motif in scaffold
#     sample[:, new_start:new_start+len(motif_seq)] = torch.tensor(motif_tokenized)
#     nonmask_locations = (sample[0] != masking_idx).nonzero().flatten()
#     new_start_idxs, new_end_idxs = [new_start], [new_start+len(motif_seq)]
#     value, loc = (sample == masking_idx).long().nonzero(as_tuple=True) # locations that need to be unmasked
#     loc = np.array(loc)
#     np.random.shuffle(loc)
#     sample = sample.long().to(device)


#     with torch.no_grad():
#         for i in loc:
#             timestep = torch.tensor([0] * batch_size)  # placeholder but not called in model
#             timestep = timestep.to(device)
#             # if random_baseline:
#             #     p_sample = torch.multinomial(torch.tensor(train_prob_dist), num_samples=1)
#             # else:
#             prediction = model(sample, timestep)
#             p = prediction[:, i, :len(tokenizer.all_aas) - 6]  # only canonical
#             p = torch.nn.functional.softmax(p, dim=1)  # softmax over categorical probs
#             p_sample = torch.multinomial(p, num_samples=1)
#             sample[:, i] = p_sample.squeeze()
#     print("Generated sequence:", [tokenizer.untokenize(s) for s in sample])
#     untokenized = [tokenizer.untokenize(s) for s in sample]

    
#     strings = []
#     start_idxs = []
#     end_idxs = []
#     scaffold_lengths = []
#     scaffold_length = random.randint(100, 1024)
#     # string, new_start_idx, new_end_idx = generate_scaffold(model, pdb_id, [motif_start],
#     #                                                         [motif_end], scaffold_length,
#     #                                                         data_dir, tokenizer, device=device)
#     strings.append(untokenized)
#     start_idxs.append(new_start_idxs)
#     end_idxs.append(new_end_idxs)
#     scaffold_lengths.append(scaffold_length)


# out_dir = os.path.join('/home/chialun/projects/evodiff/src/training/performance')
# out_fpath = os.path.join(out_dir, f'epoch{args.epochs}_lr{args.lr}_accum{args.gradient_accumulation_steps}_warmup{args.warmup_steps}_weight-decay{args.weight_decay}')
# os.makedirs(out_fpath, exist_ok=True)
# print(out_fpath)
# save_df = pd.DataFrame(list(zip(strings, start_idxs, end_idxs, scaffold_lengths)), columns=['seqs', 'start_idxs', 'end_idxs', 'scaffold_lengths'])
# save_df.to_csv(os.path.join(out_fpath, 'motif_df.csv'), index=True)



# with open(os.path.join(out_fpath,'generated_samples_string.csv'), 'w') as f:
#     for _s in strings:
#         f.write(_s[0]+"\n")
# with open(os.path.join(out_fpath,'generated_samples_string.fasta'), 'w') as f:
#     for i, _s in enumerate(strings):
#         f.write(">SEQUENCE_" + str(i) + "\n" + str(_s[0]) + "\n")


# # After cond gen, run omegafold
# print("Finished generation, starting omegafold")

# run_omegafold(out_fpath, fasta_file="generated_samples_string.fasta")

# print("Cleaning PDBs")
# # clean PDB for TMScore analysis
# clean_pdb(os.path.join(out_fpath, 'pdb/'), data_dir, pdb_id)

# print("Getting TM scores")
# # Get TMscores
# run_tmscore(out_fpath, pdb_id, num_seqs, reres=True,path_to_tmscore='/home/chialun/tools/./TMscore')# path_to_tmscore=top_dir+'TMscore/', 




