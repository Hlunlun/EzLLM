import os
import re
import pandas as pd
import pathlib
import argparse
import subprocess
from Bio import SeqIO


'''
python conditional_generation_msa.py --cond-task scaffold --pdb A0A0A0PHP9 --num-seqs 1 --start-idx 0 --end-idx 10
python conditional_generation.py --cond-task scffold --pdb A0A0A0PHP9 --num-seqs 1 --start-idx 0 --end-idx 10 
'''



def main():
    args = parse_arguments()
    home_dir = os.path.join(str(pathlib.Path.home()), 'projects/evodiff')
    file_paths = get_file_paths(home_dir, args)

    info_df = pd.read_csv(file_paths['info_file'])
    pdb_files = [file[:-6] for file in os.listdir(file_paths['pdb_dir']) if file.endswith('.fasta')]
    msa_files = [file[:-4] for file in os.listdir(file_paths['msa_dir']) if file.endswith('.a3m')]


    for pdb in pdb_files:

        try:
            cond_gen(pdb, info_df, args, file_paths)
            
            tem_sta_pro(pdb, args, file_paths, args.model_type)

            pH_predict(pdb, args, file_paths, args.model_type)
        except Exception as e:
            print(f'An error occured: {e} when processing {pdb}')

    
    # for msa in msa_files:
    #     cond_gen_msa(msa, info_df, args, file_paths)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process PDB files and run analysis.")
    parser.add_argument('--num-seqs', type=int, default=10,
                        help="Number of sequences generated per scaffold length")
    parser.add_argument('--scaffold-min', type=int, default=100,
                        help="Min scaffold length")
    parser.add_argument('--scaffold-max', type=int, default=150,
                        help="Max scaffold length, will randomly choose a value between min/max")
    parser.add_argument('--max-seq-length', type=int, default=1024,
                        help="Max sequence length to sample from IDR set")
    parser.add_argument('--max-seq-len', type=int, default=1024,
                        help="Max sequence length to sample from IDR set")
    parser.add_argument('--range', type=str, default='species',
                        help="Range for data selection")
    parser.add_argument('--model-type', type=str, default='oa_dm_640M',
                        help='Choice of: carp_38M carp_640M esm1b_650M oa_dm_38M oa_dm_640M lr_ar_38M lr_ar_640M')    
    parser.add_argument('--msa-model-type', type=str, default='msa_oa_dm_maxsub',
                        help='Choice of: msa_oa_dm_randsub, msa_oa_dm_maxsub, esm_msa_1b')
    parser.add_argument('--state-dict-path', type=str, default = 'src/training/models')
    parser.add_argument('--fine-tune', type = int, default = 0)
    parser.add_argument('--gpus', type = int, default=3)
    return parser.parse_args()



def get_file_paths(home_dir, args):
    
    data_path = os.path.join(home_dir, f'data/ec_{args.range}')
    top_model_path = get_best_model(home_dir, args)
    return {
        'home_dir': home_dir,
        'data_path': data_path,
        'cond_gen': 'conditional_generation.py',
        'cond_gen_msa': 'conditional_generation_msa.py',
        'rmsd': 'rmsd_analysis.py',
        'tem_sta_pro': '/home/chialun/projects/TemStaPro',
        'pH': '/home/chialun/projects/EpHod',
        'pdb_dir': os.path.join(data_path, 'scaffolding-pdbs'),
        'msa_dir': os.path.join(data_path, 'scaffolding-msas'),
        'data_dir': os.path.join(home_dir, 'data'),
        'info_file': os.path.join(data_path, 'motif_dict.csv'),
        'top_model_path': top_model_path,
    }

def get_best_model(home_dir, args):

    experiment_dir = os.path.join(home_dir, args.state_dict_path)
    top_model =''
    best_accu = 0
    for model_dir in os.listdir(experiment_dir):
        for model_state in model_dir:
            match = re.search(r'accu([\d.]+)', model_state)
            if match:
                model_accu = float(match.group(1))
                # Update top model if this one has the highest accuracy
                if model_accu > best_accu:
                    best_accu = model_accu
                    top_model = os.path.join(experiment_dir, model_dir, model_state)

    return top_model


def cond_gen(pdb, info_df, args, file_paths):
    motif_start, motif_end = info_df[info_df['Entry'] == pdb][['Motif_start', 'Motif_end']].values[0]
    
    common_args = [
        '--pdb', pdb,
        '--scaffold-min', str(args.scaffold_min),
        '--scaffold-max', str(args.scaffold_max),
        '--num-seqs', str(args.num_seqs),
    ]
    if motif_start != '' and motif_end != '':
        common_args.extend(['--start-idx', str(int(motif_start))])
        common_args.extend(['--end-idx', str(int(motif_end))])

    con_gen_args =[]
    if args.fine_tune == 1:        
        con_gen_args.extend(['--state-dict-path', file_paths['top_model_path']])

    subprocess.run(['python', file_paths['cond_gen'],'--gpus', str(args.gpus), '--cond-task', 'scaffold', '--max-seq-len', str(args.max_seq_len),'--data-path', f'ec_{args.range}']
                    + con_gen_args + common_args)
    subprocess.run(['python', file_paths['rmsd']] + common_args)
    


def cond_gen_msa(pdb, info_df, args, file_paths):
    motif_start, motif_end = info_df[info_df['Entry'] == pdb][['Motif_start', 'Motif_end']].values[0]

    common_args = [
        '--pdb', pdb,
        '--num-seqs', str(args.num_seqs),
        '--max-seq-len', str(args.max_seq_len)
    ]

    if motif_start != '' and motif_end != '':
        common_args.extend(['--start-idx', str(int(motif_start))])
        common_args.extend(['--end-idx', str(int(motif_end))])

    subprocess.run(['python', file_paths['cond_gen_msa'], '--cond-task', 'scaffold', '--data-path', f'ec_{args.range}'] + common_args)
    

def tem_sta_pro(pdb, args, file_paths, model_type, k_value=41):

    pdb_dir = os.path.join(file_paths['data_dir'], model_type, pdb)
    success_dir = os.path.join(pdb_dir, 'success')
    temperature_dir = os.path.join(pdb_dir, 'success', 'temperature')
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(temperature_dir, exist_ok=True)

    if os.path.exists(os.path.join(success_dir,'successes.fasta')):

        success_seq = list(SeqIO.parse(os.path.join(success_dir,'successes.fasta'), 'fasta'))

        if len(success_seq) > 0:
            common_args = [
                '-f', os.path.join(success_dir,'successes.fasta'),
                '-e', temperature_dir,
                '-d', os.path.join(file_paths['tem_sta_pro'], 'ProtTrans'),
                '-t', file_paths['tem_sta_pro'], # model path
                '--curve-smoothening', 
                '-p', temperature_dir,
                # '--per-segment-output', os.path.join(temperature_dir, f'per_segment_predictions_k{k_value}.tsv'),
                # '--per-res-output', os.path.join(temperature_dir, 'per_res_predictions_per_res.tsv'),
                '--mean-output', os.path.join(temperature_dir, 'mean_output.tsv')
            ]

            subprocess.run([os.path.join(file_paths['tem_sta_pro'], 'temstapro')] + common_args)
    




def pH_predict(pdb, args, file_paths, model_type, batch_size = 1, verbose = 1, att_weight=1, emb=1):

    pdb_dir = os.path.join(file_paths['data_dir'], model_type, pdb)
    success_dir = os.path.join(pdb_dir, 'success')
    pH_dir = os.path.join(pdb_dir, 'success', 'pH')
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(pH_dir, exist_ok=True)

    if os.path.exists(os.path.join(success_dir,'successes.fasta')):

        success_seq = list(SeqIO.parse(os.path.join(success_dir,'successes.fasta'), 'fasta'))

        if len(success_seq) > 0:
            
            common_args = [ 
                '--fasta_path',  os.path.join(success_dir,'successes.fasta'),
                '--save_dir', pH_dir,
                '--csv_name', 'ephod_pred.csv',
                '--batch_size', str(batch_size),
                '--verbose', str(verbose),
                '--save_attention_weights', str(att_weight),
                '--attention_mode', "average",
                '--save_embeddings', str(emb), 
            ]

            subprocess.run(['python', os.path.join(file_paths['pH'], 'ephod/predict.py')] + common_args)
    






if __name__ == "__main__":
    main()