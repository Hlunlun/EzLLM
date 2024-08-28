import os
from torch.utils.data import Dataset
from Bio import SeqIO
from sequence_models.utils import parse_fasta
from evodiff.utils import Tokenizer
from sequence_models.constants import PROTEIN_ALPHABET, trR_ALPHABET, PAD, GAP
import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming, cdist




class FASTADataset(Dataset):

    def __init__(self, dataset_path='/home/chialun/projects/evodiff/data/ec_species') -> None:
        super().__init__()
        
        self.data = []
        all_files = [file for file in os.listdir(os.path.join(dataset_path, 'scaffolding-pdbs')) if file.endswith('.fasta')]
        for filename in all_files:
            records = list(SeqIO.parse(os.path.join(dataset_path, 'scaffolding-pdbs', filename), 'fasta'))
            self.data.append([str(records[0].seq)])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



def read_openfold_files(data_dir, filename):
    """
    Helper function to read the openfold files

    inputs:
        data_dir : path to directory with data
        filename: MSA name

    outputs:
        path: path to .a3m file
    """
    if os.path.exists(data_dir + filename + '/a3m/uniclust30.a3m'):
        path = data_dir + filename + '/a3m/uniclust30.a3m'
    elif os.path.exists(data_dir + filename + '/a3m/bfd_uniclust_hits.a3m'):
        path = data_dir + filename + '/a3m/bfd_uniclust_hits.a3m'
    else:
        raise Exception("Missing filepaths")
    # return paths





class A3MMSADataset(Dataset):
    """Build dataset for A3M data: MSA Absorbing Diffusion model"""

    def __init__(self, selection_type, n_sequences, max_seq_len, data_dir=None, min_depth=None):
        """
        Args:
            selection_type: str,
                MSA selection strategy of random or MaxHamming
            n_sequences: int,
                number of sequences to subsample down to
            max_seq_len: int,
                maximum MSA sequence length
            data_dir: str,
                if you have a specified data directory
        """
        alphabet = PROTEIN_ALPHABET
        self.tokenizer = Tokenizer(alphabet)
        self.alpha = np.array(list(alphabet))
        self.gap_idx = self.tokenizer.alphabet.index(GAP)

        # Get npz_data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(data_dir)

        [print("Excluding", x) for x in os.listdir(self.data_dir) if x.endswith('.npz')]
        all_files = [x for x in os.listdir(self.data_dir) if not x.endswith('.npz')]
        all_files = sorted(all_files)
        print("unfiltered length", len(all_files))
        
        print(len(np.load(data_dir + 'openfold_gap_depths.npz')['arr_0']))
        print(np.load(data_dir + 'openfold_gap_depths.npz')['arr_0'])
        print(len(np.load(data_dir+'openfold_lengths.npz')['ells']))
        print(np.load(data_dir+'openfold_lengths.npz')['ells'])
        print(len(np.load(data_dir+'openfold_depths.npz')['arr_0']))
        print(np.load(data_dir+'openfold_depths.npz')['arr_0'])

        ## Filter based on depth (keep > 64 seqs/MSA)
        if not os.path.exists(data_dir + 'openfold_lengths.npz'):
            raise Exception("Missing openfold_lengths.npz in openfold/")
        if not os.path.exists(data_dir + 'openfold_depths.npz'):
            #get_msa_depth_openfold(data_dir, sorted(all_files), 'openfold_depths.npz')
            raise Exception("Missing openfold_depths.npz in openfold/")
        if min_depth is not None: # reindex, filtering out MSAs < min_depth
            _depths = np.load(data_dir+'openfold_depths.npz')['arr_0']
            depths = pd.DataFrame(_depths, columns=['depth'])
            depths = depths[depths['depth'] >= min_depth]
            keep_idx = depths.index

            _lengths = np.load(data_dir+'openfold_lengths.npz')['ells']
            lengths = np.array(_lengths)[keep_idx]
            all_files = np.array(all_files)[keep_idx]
            print("filter MSA depth > 64", len(all_files))

        # Re-filter based on high gap-contining rows
        if not os.path.exists(data_dir + 'openfold_gap_depths.npz'):
            #get_sliced_gap_depth_openfold(data_dir, all_files, 'openfold_gap_depths.npz', max_seq_len=max_seq_len)
            raise Exception("Missing openfold_gap_depths.npz in openfold/")
        _gap_depths = np.load(data_dir + 'openfold_gap_depths.npz')['arr_0']
        gap_depths = pd.DataFrame(_gap_depths, columns=['gapdepth'])
        gap_depths = gap_depths[gap_depths['gapdepth'] >= min_depth]
        filter_gaps_idx = gap_depths.index
        lengths = np.array(lengths)[filter_gaps_idx]
        all_files = np.array(all_files)[filter_gaps_idx]
        print("filter rows with GAPs > 512", len(all_files))

        self.filenames = all_files  # IDs of samples to include
        self.lengths = lengths # pass to batch sampler
        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = read_openfold_files(self.data_dir, filename)
        parsed_msa = parse_fasta(path)

        aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in parsed_msa]
        aligned_msa = [''.join(seq) for seq in aligned_msa]

        tokenized_msa = [self.tokenizer.tokenizeMSA(seq) for seq in aligned_msa]
        tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
        msa_seq_len = len(tokenized_msa[0])

        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len

        # Slice to 512
        sliced_msa_seq = tokenized_msa[:, slice_start: slice_start + self.max_seq_len]
        anchor_seq = sliced_msa_seq[0]  # This is the query sequence in MSA

        # slice out all-gap rows
        sliced_msa = [seq for seq in sliced_msa_seq if (list(set(seq)) != [self.gap_idx])]
        msa_num_seqs = len(sliced_msa)

        if msa_num_seqs < self.n_sequences:
            print("before for len", len(sliced_msa_seq))
            print("msa_num_seqs < self.n_sequences should not be called")
            print("tokenized msa shape", tokenized_msa.shape)
            print("tokenized msa depth", len(tokenized_msa))
            print("sliced msa depth", msa_num_seqs)
            print("used to set slice")
            print("msa_seq_len", msa_seq_len)
            print("self max seq len", self.max_seq_len)
            print(slice_start)
            import pdb; pdb.set_trace()
            output = np.full(shape=(self.n_sequences, seq_len), fill_value=self.tokenizer.pad_id)
            output[:msa_num_seqs] = sliced_msa
            raise Exception("msa num_seqs < self.n_sequences, indicates dataset not filtered properly")
        elif msa_num_seqs > self.n_sequences:
            if self.selection_type == 'random':
                random_idx = np.random.choice(msa_num_seqs - 1, size=self.n_sequences - 1, replace=False) + 1
                anchor_seq = np.expand_dims(anchor_seq, axis=0)
                output = np.concatenate((anchor_seq, np.array(sliced_msa)[random_idx.astype(int)]), axis=0)
            elif self.selection_type == "MaxHamming":
                output = [list(anchor_seq)]
                msa_subset = sliced_msa[1:]
                msa_ind = np.arange(msa_num_seqs)[1:]
                random_ind = np.random.choice(msa_ind)
                random_seq = sliced_msa[random_ind]
                output.append(list(random_seq))
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, (random_ind - 1), axis=0)
                m = len(msa_ind) - 1
                distance_matrix = np.ones((self.n_sequences - 2, m))

                for i in range(self.n_sequences - 2):
                    curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                    curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                    distance_matrix[i] = curr_dist
                    col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
                    max_ind = np.argmax(col_min)
                    random_ind = max_ind
                    random_seq = msa_subset[random_ind]
                    output.append(list(random_seq))
                    random_seq = np.expand_dims(random_seq, axis=0)
                    msa_subset = np.delete(msa_subset, random_ind, axis=0)
                    distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
        else:
            output = sliced_msa

        output = [''.join(seq) for seq in self.alpha[output]]
        return output