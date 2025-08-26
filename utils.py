from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from rdkit import Chem

from subword_nmt.apply_bpe import BPE
import codecs
import pandas as pd

from models.deepsynergy import *
from models.deepsynergy_causal import *

from models.deepdds import *
from models.deepdds_causal import *

from models.synergyx import *
from models.synergyx_causal import *

def get_data(args, folder_path):
    data_path = "{}/data".format(folder_path)

    drug_smiles_file = '{}/drug.csv'.format(data_path)
    drug_synergy_file = '{}/drugcomb.csv'.format(data_path)
    gene_file = '{}/expression.csv'.format(data_path)


    expression_file = '{}/expression.csv'.format(data_path)
    mutations_file = '{}/mutation.csv'.format(data_path)
    dependency_file = '{}/dependency.csv'.format(data_path)
    copynumber_file = '{}/copynumber.csv'.format(data_path)

    drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[1])
    drug2smile = dict(zip(drug['drugname'], drug['smiles']))
    drug2espf = get_espf_dict(drug, folder_path)
    drug2graph = get_graph_dict(drug, folder_path)

    gene = pd.read_csv(expression_file, sep=',', header=0, index_col=[0])
    gene_data = pd.read_csv(expression_file, sep=',', header=0, index_col=[0])
    mutation_data = pd.read_csv(mutations_file, sep=',', header=0, index_col=[0])
    dependency_data = pd.read_csv(dependency_file, sep=',', header=0, index_col=[0])
    copynumber_data = pd.read_csv(copynumber_file, sep=',', header=0, index_col=[0])

    cline_required = list(set(gene.index))
    cline_num = len(cline_required)

    cline2id = dict(zip(cline_required, range(cline_num)))
    atom_type = len(ATOM_SYMBOLS) +1

    cline2gene = {}
    cline2mutation = {}
    cline2dependency = {}
    cline2copynumber = {}

    for cline, cline_id in cline2id.items():
        cline2gene[cline_id] = np.array(gene_data.loc[cline].values, dtype='float32')
        cline2mutation[cline_id] = np.array(mutation_data.loc[cline].values, dtype='float32')
        cline2dependency[cline_id] = np.array(dependency_data.loc[cline].values, dtype='float32')
        cline2copynumber[cline_id] = np.array(copynumber_data.loc[cline].values, dtype='float32')
    gene_dim = gene_data.shape[1]
    mutation_dim = mutation_data.shape[1]
    gene_list = copynumber_data.columns

    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)
    cell_line_list = synergy_load.drop_duplicates(subset=['cell_line_name'])['cell_line_name']
    labels = args.labels
    if labels == 'sscore':
        label_index = 5
    elif labels == 'zip':
        label_index = 6
    elif labels == 'loewe':
        label_index = 7
    elif labels == 'hsa':
        label_index = 8
    elif labels == 'bliss':
        label_index = 9
    else:
        raise ValueError(f"Invalid labels: {labels}")
    tissue_name_list = list(synergy_load['tissue_name'])

    print(labels)
    print(label_index)

    synergy = [[row[1], row[2], cline2id[row[4]], float(row[label_index])] for _, row in
               synergy_load.iterrows()]

    novel_combos_file = '{}/novel_combos.csv'.format(data_path)
    novel_combos_load = pd.read_csv(novel_combos_file, sep=',', header=0)
    novel_combos = [[row[2], row[3], cline2id[row[1]], float(0)] for _, row in
               novel_combos_load.iterrows()]

    return synergy, drug2smile, drug2espf, cline2gene, cline2mutation, cline2dependency, cline2copynumber, gene_dim, mutation_dim, label_index, tissue_name_list, cell_line_list, gene_list, drug2graph, atom_type, cline2id, novel_combos


def get_espf_dict(drug, folder_path):
    drug2espf = {}
    dataFolder = '{}/info'.format(folder_path)
    vocab_path = '{}/codes_protein.txt'.format(dataFolder)
    bpe_codes_protein = codecs.open(vocab_path)
    pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
    sub_csv = pd.read_csv(dataFolder + '/subword_units_map_protein.csv')
    idx2word_p = sub_csv['index'].values
    words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))
    vocab_path = dataFolder + '/codes_drug.txt'
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    sub_csv = pd.read_csv(dataFolder + '/subword_units_map_drug.csv')
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
    ################################
    for smile in tqdm(drug['smiles'].values):
        drug2espf[smile] = get_espf(smile, dbpe, words2idx_d, idx2word_d)
    return drug2espf

def get_espf(smile, dbpe, words2idx_d, idx2word_d):
    s = smile
    t = dbpe.process_line(s).split()
    i = [words2idx_d[i] for i in t]
    v = np.zeros(len(idx2word_d), )
    tensor = torch.tensor(i)
    target_length = 50
    tensor_length = tensor.size(0)

    if tensor_length < target_length:
        padding_length = target_length - tensor_length
        padding = torch.zeros(padding_length, dtype=tensor.dtype)
        padded_tensor = torch.cat((tensor, padding))

        mask = torch.ones(tensor_length, dtype=torch.long)
        mask_padding = torch.zeros(padding_length, dtype=torch.long)
        mask = torch.cat((mask, mask_padding))
    elif tensor_length > target_length:
        padded_tensor = tensor[:target_length]

        mask = torch.ones(target_length, dtype=torch.long)
    else:
        padded_tensor = tensor
        mask = torch.ones(target_length, dtype=torch.long)
    merged_tensor = torch.stack((padded_tensor, mask), dim=0)
    return merged_tensor

def get_graph_dict(drug, folder_path):
    drug2graph = {}
    for smile in tqdm(drug['smiles'].values):
        drug2graph[smile] = smiles_to_pyg(smile)
    return drug2graph

ATOM_SYMBOLS = [
    'C', 'N', 'O', 'H', 'S', 'P', 'F', 'Cl', 'Br', 'I',
    'B', 'Si', 'Se', 'As', 'Li', 'Na', 'K', 'Mg', 'Ca',
    'Fe', 'Zn', 'Cu', 'Mn', 'Co', 'Ni', 'Pt', 'Au', 'Ag',
    'Se', 'Sn'
]

def get_atom_symbol_index(atom_symbol: str) -> int:
    if atom_symbol in ATOM_SYMBOLS:
        return ATOM_SYMBOLS.index(atom_symbol)
    else:
        return len(ATOM_SYMBOLS)  # 用作 <UNK>

def smiles_to_pyg(smiles_str: str) -> Data:
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None

    atom_indices = []
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        idx = get_atom_symbol_index(atom_symbol)
        atom_indices.append(idx)

    x = torch.tensor(atom_indices, dtype=torch.long)

    edge_list = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        edge_list.append([begin_idx, end_idx])
        edge_list.append([end_idx, begin_idx])

    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # shape = (2, E)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return [x, edge_index]

def get_atom_features(atom):
    element = atom.GetSymbol()
    elements = [
    'C', 'N', 'O', 'H', 'S', 'P', 'F', 'Cl', 'Br', 'I',
    'B', 'Si', 'Se', 'As', 'Li', 'Na', 'K', 'Mg', 'Ca',
    'Fe', 'Zn', 'Cu', 'Mn', 'Co', 'Ni', 'Pt', 'Au', 'Ag',
    'Se', 'Sn', 'Other'
]
    feature = [1 if element == e else 0 for e in elements]

    feature += [
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        int(atom.GetIsAromatic())
    ]
    return np.array(feature, dtype=np.float32)


def process_data(synergy, drug2smile, cline2expression, cline2mutation, cline2dependency, cline2copynumber):
    processed_synergy = []

    for row in synergy:
        processed_synergy.append([drug2smile[row[0]], drug2smile[row[1]],
                                cline2expression[int(row[2])],
                                  cline2copynumber[int(row[2])],
                                  cline2mutation[int(row[2])],
                                  cline2dependency[int(row[2])], float(row[3])])

    return np.array(processed_synergy, dtype=object)
def synergy_data_split(synergy_df, args):
    source_data = np.array(synergy_df)

    # 源数据划分
    source_train, source_temp = train_test_split(
        source_data,
        test_size=0.2,
        random_state=args.split_seed
    )
    source_val, source_test = train_test_split(
        source_temp,
        test_size=0.5,
        random_state=args.split_seed
    )

    #

    return source_train, source_val, source_test

def get_model(gene_dim, args):

    if args.model == 'deepsynergy':
        model = deepsynergy(gene_dim, args=args)
    if args.model == 'deepsynergy_causal':
        model = deepsynergy_causal(gene_dim, args=args)
    
    if args.model == 'synergyx':
        model = synergyx(gene_dim = gene_dim, args=args)
    if args.model == 'synergyx_causal':
        model = synergyx_causal(gene_dim = gene_dim, args=args)
    if args.model == 'deepdds':
        model = deepdds(gene_dim = gene_dim, args=args)
    if args.model == 'deepdds_causal':
        model = deepdds_causal(gene_dim = gene_dim, args=args)

    return model

def find_drug_combinations(df, args):
    drug1 = args.IG_drugA.lower()
    drug2 = args.IG_drugB.lower()

    df_search = df.copy()
    df_search['drugA'] = df_search['drugA'].str.lower()
    df_search['drugB'] = df_search['drugB'].str.lower()

    condition = (
            ((df_search['drugA'] == drug1) & (df_search['drugB'] == drug2)) |
            ((df_search['drugA'] == drug2) & (df_search['drugB'] == drug1))
    )
    return df[condition]