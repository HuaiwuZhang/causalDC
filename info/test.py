import torch
from subword_nmt.apply_bpe import BPE
import codecs
import pandas as pd
import numpy as np

dataFolder = 'C:/work/phd/project/self_cade_baseline_test/synergyx/ESPF/info'

# For Proteins
vocab_path = dataFolder + '/codes_protein.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv(dataFolder + '/subword_units_map_protein.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

# For Drugs
vocab_path = dataFolder + '/codes_drug.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv(dataFolder + '/subword_units_map_drug.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

# Example: Given Drug SMILES String s, output a bit vector v
s = 'CC(C)C(=C)CC(O)C(C)(O)[C@H]1CC[C@H]2C3=C[C@H](OC(=O)C)[C@H]4[C@@H](OC(=O)C)[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C'
t = dbpe.process_line(s).split()
# t: 'CC(C)C (=C )CC (O)C (C) (O) [C@H]1CC [C@H]2C 3=C [C@H](OC(=O)C) [C@H]4 [C@@H](OC(=O)C) [C@@H](O)CC [C@]4(C)[C@H]3CC [C@]12C'
i = [words2idx_d[i] for i in t]
v = np.zeros(len(idx2word_d), )
i = torch.tensor(i)
v[i] = 1