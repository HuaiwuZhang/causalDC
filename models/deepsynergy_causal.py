import torch
from torch import nn

from src.base.model import BaseModel

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def drug_feat(drug_subs_codes, device, patch, length):
    v = drug_subs_codes
    subs = v[:, 0].long().to(device)
    subs_mask = v[:, 1].long().to(device)

    if patch > length:
        padding = torch.zeros(subs.size(0), patch - length).long().to(device)
        subs = torch.cat((subs, padding), 1)
        subs_mask = torch.cat((subs_mask, padding), 1)

    expanded_subs_mask = subs_mask.unsqueeze(1).unsqueeze(2)
    expanded_subs_mask = (1.0 - expanded_subs_mask) * -10000.0

    return subs, expanded_subs_mask.float()

class MultiLevelProjection(nn.Module):
    def __init__(self, dims=[14749, 1500, 150], feat_dim=4, hidden_dim=128):
        super().__init__()
        self.proj_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.proj_layers.append(
                nn.Linear(dims[i], dims[i + 1], bias=False)
            )
        self.feat_transform = nn.Linear(feat_dim, hidden_dim)

    def forward(self, x):
        x_trans = self.feat_transform(x)

        projected = x_trans.transpose(1,2)
        for layer in self.proj_layers:
            projected = layer(projected)
        return projected.transpose(1,2)

class CausalMask(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mask_generator = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        mask_logits = self.mask_generator(x)
        return torch.softmax(mask_logits, dim=-1)

class deepsynergy_causal(BaseModel):
    def __init__(self, gene_dim,
                 **args
                 ):
        super(deepsynergy_causal, self).__init__(**args)

        self.patch = 150
        self.max_length = 50
        input_dim_drug = 2586
        hidden_size = 128
        hidden_dropout_prob = 0.1
        self.genes_nums = 14749
        total_dim = 3*150*128
        self.drug_emb = Embeddings(input_dim_drug, hidden_size, self.patch, hidden_dropout_prob)
        self.cell_projector = MultiLevelProjection()
        self.causal_mask_layer = CausalMask()
        self.mlp_c = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1)
        )

        self.mlp_t = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1)
        )


    def forward(self, drugA, drugB, cline_gene, cline_mutation, cline_dependency, cline_copynumber):
        batch_size = drugA.size(0)
        device = 'cuda:0'
        drugA, drugA_attention_mask = drug_feat(drugA, device, self.patch, self.max_length)
        drugB, drugB_attention_mask = drug_feat(drugB, device, self.patch, self.max_length)

        drugA = self.drug_emb(drugA)
        drugB = self.drug_emb(drugB)
        drugA = drugA.float()
        drugB = drugB.float()

        cline_gene = cline_gene.unsqueeze(-1)
        cline_mutation = cline_mutation.unsqueeze(-1)
        cline_dependency = cline_dependency.unsqueeze(-1)
        cline_copynumber = cline_copynumber.unsqueeze(-1)
        x_cell = torch.cat((cline_gene, cline_mutation, cline_dependency, cline_copynumber), dim=-1)

        x_cell = x_cell.type(torch.float32)
        cell1 = x_cell.view(batch_size, self.genes_nums, -1)
        cell1 = self.cell_projector(cell1).float()

        cell_mask = self.causal_mask_layer(cell1)
        cell1_c = cell_mask[:, :, 0].unsqueeze(-1) * cell1
        cell1_t = cell_mask[:, :, 1].unsqueeze(-1) * cell1

        drugA = drugA.reshape(drugA.size(0), -1)
        drugB = drugB.reshape(drugB.size(0), -1)
        cell1_c = cell1_c.reshape(cell1_c.size(0), -1)
        cell1_t = cell1_t.reshape(cell1_t.size(0), -1)
        if self.args.get_cell_embedding:
            cell_embedding = cell1_c
            return cell_embedding

        combined_c = torch.cat([drugA, drugB, cell1_c], dim=1)
        combined_t = torch.cat([drugA, drugB, cell1_t], dim=1)

        output_c = self.mlp_c(combined_c)
        output_t = self.mlp_t(combined_t)


        return output_c, output_t
