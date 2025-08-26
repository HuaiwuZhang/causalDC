import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch


class DrugCombDataset(Dataset):
    def __init__(self, smiles1, smiles2, expression, mutation, dependency, copynumber, labels, drug2espf):
        self.labels = labels
        self.length = len(self.labels)
        self.smiles1 = smiles1
        self.smiles2 = smiles2

        self.drug2espf = drug2espf

        self.cline_expression = expression
        self.cline_mutation = mutation
        self.cline_dependency = dependency
        self.cline_copynumber = copynumber

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.labels[idx]

        cline_expression = torch.FloatTensor(self.cline_expression[idx])
        cline_mutation = torch.FloatTensor(self.cline_mutation[idx])
        cline_dependency = torch.FloatTensor(self.cline_dependency[idx])
        cline_copynumber = torch.FloatTensor(self.cline_copynumber[idx])
        drug1_espf = self.drug2espf[self.smiles1[idx]].type(torch.int)
        drug2_espf = self.drug2espf[self.smiles2[idx]].type(torch.int)
        return drug1_espf, drug2_espf, cline_expression, cline_mutation, cline_dependency, cline_copynumber, torch.FloatTensor([label])

def create_loader(data, args, drug2espf, shuffle=False, drop_last = False):
    batch_size = args.batch_size
    return DataLoader(
        DrugCombDataset(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            data[:, 4],
            data[:, 5],
            data[:, 6],
            drug2espf
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

def create_infer_loader(data, args, drug2espf, infer_batch_size, shuffle=False, drop_last = False):
    batch_size = infer_batch_size
    return DataLoader(
        DrugCombDataset(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            data[:, 4],
            data[:, 5],
            data[:, 6],
            drug2espf
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

class DrugCombGraphDataset(Dataset):
    def __init__(self, smiles1, smiles2, expression, mutation, dependency, copynumber, labels, drug2espf, drug2graph):
        self.labels = labels
        self.length = len(self.labels)
        self.smiles1 = smiles1
        self.smiles2 = smiles2

        self.drug2espf = drug2espf
        self.drug2graph = drug2graph

        self.cline_expression = expression
        self.cline_mutation = mutation
        self.cline_dependency = dependency
        self.cline_copynumber = copynumber

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.labels[idx]

        cline_expression = torch.FloatTensor(self.cline_expression[idx])
        cline_mutation = torch.FloatTensor(self.cline_mutation[idx])
        cline_dependency = torch.FloatTensor(self.cline_dependency[idx])
        cline_copynumber = torch.FloatTensor(self.cline_copynumber[idx])
        drug1_espf = self.drug2espf[self.smiles1[idx]].type(torch.int)
        drug2_espf = self.drug2espf[self.smiles2[idx]].type(torch.int)
        drug1_graph_list = self.drug2graph[self.smiles1[idx]]
        drug2_graph_list = self.drug2graph[self.smiles2[idx]]
        drug1_graph = Data(x=drug1_graph_list[0], edge_index=drug1_graph_list[1])
        drug2_graph = Data(x=drug2_graph_list[0], edge_index=drug2_graph_list[1])
        label = torch.FloatTensor([label]).unsqueeze(-1)
        return {
            'drug1_espf': drug1_espf,
            'drug2_espf': drug2_espf,
            'drug1_graph': drug1_graph,
            'drug2_graph': drug2_graph,
            'expression': cline_expression,
            'mutation': cline_mutation,
            'dependency': cline_dependency,
            'copynumber': cline_copynumber,
            'label': label
        }


class DrugCombCollator:

    def __init__(self, use_graphs=True):
        self.use_graphs = use_graphs

    def __call__(self, batch):
        drug1_espf = torch.stack([item['drug1_espf'] for item in batch])
        drug2_espf = torch.stack([item['drug2_espf'] for item in batch])
        expression = torch.stack([item['expression'] for item in batch])
        mutation = torch.stack([item['mutation'] for item in batch])
        dependency = torch.stack([item['dependency'] for item in batch])
        copynumber = torch.stack([item['copynumber'] for item in batch])
        labels = torch.cat([item['label'] for item in batch])
        batch_data = {
            'drug1_espf': drug1_espf,
            'drug2_espf': drug2_espf,
            'expression': expression,
            'mutation': mutation,
            'dependency': dependency,
            'copynumber': copynumber,
            'labels': labels
        }
        if self.use_graphs:
            drug1_graphs = [item['drug1_graph'] for item in batch]
            drug2_graphs = [item['drug2_graph'] for item in batch]

            batch_data['drug1_graph'] = Batch.from_data_list(drug1_graphs)
            batch_data['drug2_graph'] = Batch.from_data_list(drug2_graphs)
        aa = torch.FloatTensor(batch_data['labels'])
        return [batch_data['drug1_graph'], batch_data['drug2_graph'], batch_data['expression'], batch_data['mutation'], batch_data['dependency'], batch_data['copynumber'], torch.FloatTensor(batch_data['labels'])]

def create_graph_loader(data, args, drug2espf, drug2graph, shuffle=False, drop_last = False):
    batch_size = args.batch_size
    dataset = DrugCombGraphDataset(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            data[:, 4],
            data[:, 5],
            data[:, 6],
            drug2espf,
            drug2graph,

        )
    collator = DrugCombCollator()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collator
    )

def create_graph_infer_loader(data, args, drug2espf, drug2graph, infer_batch_size, shuffle=False, drop_last = False):
    batch_size = infer_batch_size
    dataset = DrugCombGraphDataset(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        data[:, 4],
        data[:, 5],
        data[:, 6],
        drug2espf,
        drug2graph,

    )
    collator = DrugCombCollator()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collator
    )