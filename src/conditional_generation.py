import evodiff
from evodiff.pretrained import OA_DM_640M, OA_DM_38M, CARP_640M, LR_AR_38M, LR_AR_640M
import numpy as np
import argparse
import urllib.request
import torch
import os
import esm.inverse_folding
# from evodiff.utils import Tokenizer, run_omegafold, clean_pdb, run_tmscore #, wrap_dr_bert, read_dr_bert_output
from utils import Tokenizer, run_omegafold, clean_pdb, run_tmscore #, wrap_dr_bert, read_dr_bert_output
import pathlib
from sequence_models.utils import parse_fasta
from tqdm import tqdm
import pandas as pd
import random
from evodiff.plot import aa_reconstruction_parity_plot, idr_parity_plot



def main():

    # set seeds
    _ = torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='oa_dm_640M',
                        help='Choice of: carp_38M carp_640M esm1b_650M \
                              oa_dm_38M oa_dm_640M lr_ar_38M lr_ar_640M')
    parser.add_argument('--gpus', type=int, default=3)
    parser.add_argument('--cond-task', type=str, default='scaffold',
                        help="Choice of 'scaffold' or 'idr'")
    parser.add_argument('--pdb', type=str, default=None,
                        help="If using cond-task=scaffold, provide a PDB code and motif indexes")
    parser.add_argument('--start-idxs', type=int, action='append',
                        help="If using cond-task=scaffold, provide start and end indexes for motif being scaffolded\
                             If defining multiple motifs, supply the start and end -idx motif as a new argument\
                              ex: --start-idx 3 --end-idx 10 --start-idx 20 --end-idx 25\
                              indexes are inclusive of both start and end values.\
                              WARNING: PDBs are OFTEN indexed at a number that is not 0. If your PDB file begins at 4\
                              and the motif you want to query is residues 5 to 10, as defined by the PDB, your inputs to\
                              this code should be --start-idx 1 and --end-idx 6")
    parser.add_argument('--end-idxs', type=int, action='append')
    parser.add_argument('--num-seqs', type=int, default=10,
                        help="Number of sequences generated per scaffold length")
    parser.add_argument('--chain', type=str, default='A',
                        help="chain in PDB")
    parser.add_argument('--scaffold-min', type=int, default=100,
                        help="Min scaffold len ")
    parser.add_argument('--scaffold-max', type=int, default=150,
                        help="Max scaffold len, will randomly choose a value between min/max")
    parser.add_argument('--max-seq-length', type=int, default=1022,
                        help="Max sequence length to sample from IDR set")
    parser.add_argument('--data-path', type=str, default = 'ec_species')
    parser.add_argument('--single-res-domain', action='store_true', 
                        help="if start-idx = end-idx make sure to use single-res-domain flag or else you will get errors")
    parser.add_argument('--random-baseline', action='store_true') # for scaffold
    parser.add_argument('--state-dict-path', type=str, default = '')
    args = parser.parse_args()



    if args.cond_task == 'scaffold':
        args.start_idxs.sort()
        args.end_idxs.sort()

    # if args.random_baseline:
    #     args.model_type = 'oa_dm_640M' # placeholder

    print("USING MODEL", args.model_type)
    if args.model_type == 'oa_dm_38M':
        checkpoint = OA_DM_38M()
    elif args.model_type == 'oa_dm_640M':
        checkpoint = OA_DM_640M()
    elif args.model_type == 'carp_640M':
        checkpoint = CARP_640M()
    elif args.model_type == 'lr_ar_38M':
        checkpoint = LR_AR_38M()
    elif args.model_type == 'lr_ar_640M':
        checkpoint = LR_AR_640M()
    else:
        raise Exception("Please select either oa_dm_38M, oa_dm_640M, carp_640M, lr_ar_38M, or lr_ar_640M. You selected: ", args.model_type, ". If you want to generate a random baseline, add the --random-baseline flag to any model.")

    model, collater, tokenizer, scheme = checkpoint
    model.eval().cuda()

    ''' use fine-tune model '''
    if args.state_dict_path != '':
        model.load_state_dict(torch.load(args.state_dict_path))

    torch.cuda.set_device(args.gpus)
    device = torch.device('cuda:' + str(args.gpus))

    model.to(device) # all tensor on same device 20240718 --LUN

    home = str(pathlib.Path.home()) +'/projects/evodiff/data/'
    top_dir = home

    out_fpath = os.path.join(home, args.model_type, args.pdb) 



    if not os.path.exists(out_fpath):
        os.makedirs(out_fpath)
    if not os.path.exists(os.path.join(out_fpath,'plots')):
        os.makedirs(os.path.join(out_fpath,'plots'))
        os.makedirs(os.path.join(out_fpath,'plots/svg/'))
        os.makedirs(os.path.join(out_fpath,'fasta/'))
        os.makedirs(os.path.join(out_fpath,'fasta/plots/'))



    data_top_dir = os.path.join(top_dir, args.data_path)

    if args.cond_task == 'scaffold':
        strings = []
        start_idxs = []
        end_idxs = []
        scaffold_lengths = []
        for i in range(args.num_seqs):
            scaffold_length = random.randint(args.scaffold_min, args.scaffold_max)
            if args.model_type == 'oa_dm_38M' or args.model_type == 'oa_dm_640M' or args.model_type == 'carp_38M'\
                    or args.model_type == 'carp_640M':
                string, new_start_idx, new_end_idx = generate_scaffold(model, args.pdb, args.start_idxs,
                                                                               args.end_idxs, scaffold_length,
                                                                               data_top_dir, tokenizer, device=device,
                                                                               random_baseline=args.random_baseline,
                                                                               single_res_domain=args.single_res_domain,
                                                                               chain=args.chain)
            elif args.model_type == 'lr_ar_38M' or args.model_type == 'lr_ar_640M':
                string, new_start_idx, new_end_idx = generate_autoreg_scaffold(model, args.pdb, args.start_idxs,
                                                                               args.end_idxs, scaffold_length,
                                                                               data_top_dir, tokenizer, device=device,
                                                                               single_res_domain=args.single_res_domain,
                                                                               chain=args.chain)
            strings.append(string)
            start_idxs.append(new_start_idx)
            end_idxs.append(new_end_idx)
            scaffold_lengths.append(scaffold_length)


        save_df = pd.DataFrame(list(zip(strings, start_idxs, end_idxs, scaffold_lengths)), columns=['seqs', 'start_idxs', 'end_idxs', 'scaffold_lengths'])
        save_df.to_csv(os.path.join(out_fpath, 'motif_df.csv'), index=True)

    with open(os.path.join(out_fpath,'generated_samples_string.csv'), 'w') as f:
        for _s in strings:
            f.write(_s[0]+"\n")
    with open(os.path.join(out_fpath,'generated_samples_string.fasta'), 'w') as f:
        for i, _s in enumerate(strings):
            f.write(">SEQUENCE_" + str(i) + "\n" + str(_s[0]) + "\n")



    if args.cond_task == 'scaffold':
        # After cond gen, run omegafold
        print("Finished generation, starting omegafold")
        
        run_omegafold(out_fpath, fasta_file="generated_samples_string.fasta")

        print("Cleaning PDBs")
        # clean PDB for TMScore analysis
        clean_pdb(os.path.join(out_fpath, 'pdb/'), data_top_dir, args.pdb)

        print("Getting TM scores")
        # Get TMscores
        run_tmscore(out_fpath, args.pdb, args.num_seqs, reres=True,path_to_tmscore='/home/chialun/tools/./TMscore')# path_to_tmscore=top_dir+'TMscore/', 


def download_pdb(PDB_ID, outfile):
    "return PDB file from database online"
    if os.path.exists(outfile):
        print("ALREADY DOWNLOADED")
    else:
        # url = 'https://files.rcsb.org/dcd ownload/'+str(PDB_ID)+'.pdb'
        # print("DOWNLOADING PDB FILE FROM", url)
        # urllib.request.urlretrieve(url, outfile)       

        # alphafold pdb
        url = f"https://alphafold.ebi.ac.uk/files/AF-{PDB_ID}-F1-model_v4.pdb"        
        print("DOWNLOADING PDB FILE FROM", url)
        urllib.request.urlretrieve(url, outfile)


def get_motif(PDB_ID, start_idxs, end_idxs, data_top_dir='../data', chain='A'):
    "Get motif of sequence from PDB code"
    pdb_path = os.path.join(data_top_dir, f'scaffolding-pdbs/{PDB_ID}.pdb')
    print(pdb_path)
    download_pdb(PDB_ID, pdb_path)
    print("CLEANING PDB")
    clean_pdb(os.path.join(data_top_dir, 'scaffolding-pdbs/'), data_top_dir, PDB_ID)
    pdb_clean_path = os.path.join(data_top_dir,  f'scaffolding-pdbs/{PDB_ID}_clean.pdb')

    chain_ids = [chain]
    print("WARNING: USING CHAIN", chain, "FROM PDB FILE")
    structure = esm.inverse_folding.util.load_structure(pdb_clean_path, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    sequence = native_seqs[chain_ids[0]]
    print("sequence extracted from pdb", sequence)
    with open(os.path.join(data_top_dir,f'scaffolding-pdbs/{PDB_ID}.fasta'), 'a') as f:
        f.write('>' + PDB_ID+'\n'+sequence)
    print("sequence length", len(sequence))
    assert len(start_idxs) == len(end_idxs)

    end_idxs = [i+1 for i in end_idxs] # inclusive of final residue
    if len(start_idxs) > 1:
        motif = ''
        spacers = []
        # print("start idxs", start_idxs)
        # print("end idxs", end_idxs)
        for i in range(len(start_idxs)):
            motif += sequence[start_idxs[i]:end_idxs[i]]
            if i < (len(start_idxs)-1):
                spacer = start_idxs[i+1] - end_idxs[i]
                motif += '#' * spacer
                spacers.append(spacer)
    else:
        motif = sequence[start_idxs[0]: end_idxs[0]]
        spacers=[0]
    print("motif extracted from indexes supplied:", motif)
    return motif

def get_intervals(list, single_res_domain=False):
    "Given a list (Tensor) of non-masked residues get new start and end index for motif placed in scaffold"
    if single_res_domain:
        start = [l.item() for l in list]
        stop = start
    else:
        start = []
        stop = []
        for i, item in enumerate(list):
            if i == 0:
                start.append(item.item())
            elif i == (len(list)-1):
                stop.append(item.item())
            elif i != len(list) and (item+1) != list[i+1]:
                stop.append(item.item())
                start.append(list[i+1].item())
    return start, stop


def generate_scaffold(model, PDB_ID, motif_start_idxs, motif_end_idxs, scaffold_length, data_top_dir, tokenizer,
                      batch_size=1, device='gpu', random_baseline=False, single_res_domain=False, chain='A'):
    if random_baseline:
        train_prob_dist = aa_reconstruction_parity_plot(data_top_dir+'../', 'reference/', 'placeholder.csv', gen_file=False)
    mask = tokenizer.mask_id

    motif_seq = get_motif(PDB_ID, motif_start_idxs, motif_end_idxs, data_top_dir=data_top_dir, chain=chain)
    motif_tokenized = tokenizer.tokenize((motif_seq,))

    # Create input motif + scaffold
    seq_len = scaffold_length + len(motif_seq)
    sample = torch.zeros((batch_size, seq_len)) + mask # start from all mask
    new_start = np.random.choice(scaffold_length) # randomly place motif in scaffold
    sample[:, new_start:new_start+len(motif_seq)] = torch.tensor(motif_tokenized)
    nonmask_locations = (sample[0] != mask).nonzero().flatten()
    new_start_idxs, new_end_idxs = get_intervals(nonmask_locations, single_res_domain=single_res_domain)
    #print(new_start_idxs, new_end_idxs)
    value, loc = (sample == mask).long().nonzero(as_tuple=True) # locations that need to be unmasked
    loc = np.array(loc)
    np.random.shuffle(loc)
    sample = sample.long().to(device)
    with torch.no_grad():
        for i in loc:
            timestep = torch.tensor([0] * batch_size)  # placeholder but not called in model
            timestep = timestep.to(device)
            if random_baseline:
                p_sample = torch.multinomial(torch.tensor(train_prob_dist), num_samples=1)
            else:
                prediction = model(sample, timestep)
                p = prediction[:, i, :len(tokenizer.all_aas) - 6]  # only canonical
                p = torch.nn.functional.softmax(p, dim=1)  # softmax over categorical probs
                p_sample = torch.multinomial(p, num_samples=1)
            sample[:, i] = p_sample.squeeze()
    print("Generated sequence:", [tokenizer.untokenize(s) for s in sample])
    untokenized = [tokenizer.untokenize(s) for s in sample]

    return untokenized, new_start_idxs, new_end_idxs

def generate_autoreg_scaffold(model, PDB_ID, motif_start_idxs, motif_end_idxs, scaffold_length, data_top_dir, tokenizer,
                      batch_size=1, device='gpu', single_res_domain=False, chain='A'):
    mask = tokenizer.mask_id # placeholder to calculate indices here
    start = tokenizer.start_id
    stop = tokenizer.stop_id

    motif_seq = get_motif(PDB_ID, motif_start_idxs, motif_end_idxs, data_top_dir=data_top_dir, chain=chain)
    motif_tokenized = tokenizer.tokenize((motif_seq,))

    # Create input motif + scaffold (as reference for gen task)
    seq_len = scaffold_length + len(motif_seq)
    max_seq_len = seq_len
    sample_ref = torch.zeros((batch_size, seq_len)) + mask # start from all mask
    new_start = np.random.choice(scaffold_length) # randomly place motif in scaffold
    sample_ref[:, new_start:new_start+len(motif_seq)] = torch.tensor(motif_tokenized)
    nonmask_locations = (sample_ref[0] != mask).nonzero().flatten()
    new_start_idxs, new_end_idxs = get_intervals(nonmask_locations, single_res_domain=single_res_domain)
    print(new_start_idxs, new_end_idxs)
    value, loc = (sample_ref == mask).long().nonzero(as_tuple=True) # locations that need to be unmasked
    loc = np.array(loc)
    sample_ref = sample_ref.to(torch.long).to(device)

    # Start from START token
    sample = (torch.zeros((1)) + start).unsqueeze(0)  # add batch dim
    sample = sample.to(torch.long)
    sample = sample.to(device)
    reach_stop = False  # initialize
    timestep = torch.tensor([0] * batch_size)  # placeholder but not called in model
    timestep = timestep.to(device)
    max_token = stop
    with torch.no_grad():
        i = 0
        while i < max_seq_len:
            #print(i)
            if i > new_end_idxs[-1]: # Force it to continue predicting until you reach the end of motif
                max_token = len(tokenizer.alphabet)
            if reach_stop == False:
                if i in new_start_idxs: # if index is a new start idx
                    i_index = new_start_idxs.index(i)
                    # Take sample and slice in motif (preserving correct index)
                    sliced_motif = sample_ref[:, new_start_idxs[i_index]:new_end_idxs[i_index]+1]
                    i += (len(sliced_motif[0]))
                    sample = torch.cat((sample, sliced_motif), dim=1)
                    print(tokenizer.untokenize(sample[0]))
                else:  # Add residues until it predicts STOP token or hits max seq len
                    prediction = model(sample, timestep)  # , input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
                    p = prediction[:, -1, :max_token]  # predict next token
                    p = torch.nn.functional.softmax(p, dim=1)  # softmax over categorical probs
                    p_sample = torch.multinomial(p, num_samples=1)
                    sample = torch.cat((sample, p_sample), dim=1)
                    #print(tokenizer.untokenize(sample[0]))
                    # print(p_sample, stop)
                    if p_sample == stop:
                        reach_stop = True
                    i += 1
            else:
                break
    print("new sequence", [tokenizer.untokenize(s) for s in sample[:,1:-1]]) # dont need start/stop tokens
    untokenized = [tokenizer.untokenize(s) for s in sample[:,1:-1]]

    return untokenized, new_start_idxs, new_end_idxs


if __name__ == '__main__':
    main()

    













