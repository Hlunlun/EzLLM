import os
import re
import glob
import torch
import pandas as pd
from Bio import SeqIO
from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import configparser
from utils import *
from huggingface_hub import HfApi, HfFolder
from huggingface_hub import login
from evodiff.pretrained import OA_DM_640M
from huggingface_hub import Repository

''' Create dataset to Hugging Face Hub'''


def main():
    root = '../../data'
    data_dir = os.path.join(root, 'ec_species/')
    paper_file = os.path.join(root, 'uniprot/research/detergent_enzymes_from_paper/detergents.xlsx')
    motif_dir = os.path.join(root, 'uniprot/research/species/motif/all_data')
    model_dir = os.path.join('/home/chialun/projects/evodiff/src/training/models/')

    # detergent-enzyme 
    # dataset_dict = create_hf_dataset(data_dir)
    # save_dataset(dataset_dict, root, 'detergent-enzyme')

    # detergent-papers
    # dataset_dict = create_paper_dataset(paper_file)
    # save_dataset(dataset_dict, root, 'detergent-papers')

    # detergent-motif
    # dataset_dict = create_motif_dataset(motif_dir)
    # save_dataset(dataset_dict, root, 'detergent-motif')

    # detergent-models
    # save_model(model_dir, 'detergent-model')

    

def read_pdb(file_path):
    with open(file_path, 'r') as file:
        pdb_data = file.read()
    return pdb_data


def create_dataset(dir):

    data_dir = os.path.join(dir, 'scaffolding-pdbs')
    info_file = os.path.join(dir, 'motif_dict.csv')
    
    info_df = pd.read_csv(info_file)
    info_df = info_df.drop(columns=['Unnamed: 0'])
    info_df.to_csv(info_file)


    data = []
    for fasta_file in os.listdir(data_dir):
        if fasta_file.endswith('.fasta'):
            fasta_path = os.path.join(data_dir, fasta_file)
            pdb_file = fasta_file.replace('.fasta', '.pdb')
            pdb_path = os.path.join(data_dir, pdb_file)

            if os.path.exists(pdb_path):
                
                pdb_data = read_pdb(pdb_path)
                record = list(SeqIO.parse(fasta_path, 'fasta'))[0]
                id = re.search(r'\|([^\|]+)\|', str(record.id)).group(1)
                row = info_df[info_df['Entry'] == id]
                if not row.empty:
                    row = row.iloc[0]

                # ec = ec_check.values[0] if len(ec_check)>0 else ''
                
                    data.append({                        
                        'id': id,
                        'ec':row['EC'],
                        'protein': row['Protein'],
                        'species': row['Species'],
                        'organism': row['Organism'],
                        'motif_start': row['Motif_start'],
                        'motif_end': row['Motif_end'],
                        'pH_left': row['pH_left'],
                        'pH_right': row['pH_right'],
                        'temp_left': row['Temperature_left'],
                        'temp_right': row['Temperature_right'],
                        'sequence': str(record.id) + '\n' + str(record.seq),
                        'pdb_data': pdb_data
                    })
    return data


def create_hf_dataset(dir):
    data = create_dataset(dir)    
    # Convert data to DataFrame
    df = pd.DataFrame(data)   
    
    # Split into 80/20 train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # remove ['__index_level_0__']
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Convert to Hugging Face Dataset
    dataset_train = Dataset.from_pandas(train_df)
    dataset_test = Dataset.from_pandas(test_df)
    
    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })    
    return dataset_dict



def create_paper_dataset(file_path):
    xls = pd.ExcelFile(file_path)
    dataset_dict = {}
    for sheet_name in xls.sheet_names:
        if sheet_name not in ENZYME_TYPES:
            continue
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df['Temperature Optimum'] = df['Temperature Optimum'].astype(float)
        dataset_dict[sheet_name] = Dataset.from_pandas(df)
    
    dataset = DatasetDict(dataset_dict)

    return dataset


def create_motif_dataset(dir):
    csv_files = glob.glob(os.path.join(dir, '*.csv'))


    # combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    # dataset_dict = {
    #     "all": Dataset.from_pandas(combined_df)
    # }

    dataset_dict = {}
    
    for file in csv_files:
        ec_number = file.split('_')[-2]  
        df = pd.read_csv(file)
        df['Motif_start'] = df['Motif_start'].astype(float)
        df['Motif_end'] = df['Motif_end'].astype(float)
        dataset_dict[ec_number] = Dataset.from_pandas(df)


    dataset = DatasetDict(dataset_dict)

    return dataset


def save_dataset(dataset, root, repo_name):
    # Save dataset locally
    dataset.save_to_disk(os.path.join(root, repo_name))
    
    # Push to hugging face hub
    config = configparser.ConfigParser()
    config.read('/home/chialun/projects/evodiff/secrets.ini')
    api_token = config['Tokens']['HF_TOKEN']   
    
    dataset.push_to_hub(repo_name, token=api_token)   

    # Test
    print(f'Load dataset from {repo_name}')
    ds = load_dataset(f'lun610200/{repo_name}')
    # ds = load_from_disk(os.path.join(root, 'detergent-compatible'))
    print(ds)


def save_model(model_dir, repo_name):
    # Push to hugging face hub
    config = configparser.ConfigParser()
    config.read('/home/chialun/projects/evodiff/secrets.ini')
    api_token = config['Tokens']['HF_TOKEN']  
    login(token=api_token)

    checkpoint = OA_DM_640M()
    model, collater, tokenizer, scheme = checkpoint
    model.load_state_dict(torch.load('/home/chialun/projects/evodiff/src/training/models/OA_DM_640M/model_epoch57_accu63.15342603251338.pth'))

    # api = HfApi()
    # # api.create_repo(repo_name, token=api_token)
    # api.upload_folder(
    #     repo_id=repo_name,
    #     folder_path="/home/chialun/projects/evodiff/src/training/models/OA_DM_640M",
    #     token=api_token,
    #     repo_type="model"
    # )

    # tokenizer.save_pretrained('/home/chialun/projects/evodiff/src/training/models/test')
    weight = torch.load('/home/chialun/projects/evodiff/src/training/models/OA_DM_640M/model_epoch57_accu63.15342603251338.pth')
    # weight.save_pretrianed('/home/chialun/projects/evodiff/src/training/models/test')
    torch.save(weight, '/home/chialun/projects/evodiff/src/training/models/OA_DM_640M/pytorch_model.bin')



if __name__ == '__main__':
    main()
    
