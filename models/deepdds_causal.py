import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp
from src.base.model import BaseModel



class NodeFeatureEmbedder(nn.Module):

    def __init__(self, embedding_dim=64, hidden_dim=128):
        super(NodeFeatureEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(35, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, num_nodes, _ = original_shape
            x = x.view(-1, 35)

        embedded = self.mlp(x)

        return embedded

class CausalMask(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mask_generator = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        mask_logits = self.mask_generator(x)
        return torch.softmax(mask_logits, dim=-1)

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

class deepdds_causal(BaseModel):
    def __init__(self,gene_dim, n_output=1, n_filters=32, embed_dim=128,num_features_xd=64, num_features_xt=954, output_dim=19200, dropout=0.2, **args):

        super(deepdds_causal, self).__init__(**args)
        self.genes_nums = 14749

        self.node_embedder = nn.Embedding(
            num_embeddings=self.args.atom_type,
            embedding_dim=num_features_xd
        )

        self.relu = nn.ReLU()
        self.prelu = nn.PReLU(num_parameters=1, init=0.75)
        self.dropout = nn.Dropout(dropout)

        self.n_output = n_output
        self.drug1_attn1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.drug1_conv1 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.drug1_conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.drug1_conv3 = GCNConv(num_features_xd*10, num_features_xd*20)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*20, num_features_xd*40)
        self.drug1_fc_g2 = torch.nn.Linear(num_features_xd*40, output_dim)
        self.cell_projector = MultiLevelProjection()
        self.causal_mask_layer = CausalMask()
        total_dim = 3 * output_dim
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

        # DL cell featrues

    def forward(self, drugA, drugB, cline_gene, cline_mutation, cline_dependency, cline_copynumber):
        batch_size = cline_gene.size(0)

        x1, edge_index1, batch1 = drugA.x, drugA.edge_index, drugA.batch
        x2, edge_index2, batch2 = drugB.x, drugB.edge_index, drugB.batch

        # deal drug1
        x1 = self.node_embedder(x1)
        x1 = self.drug1_attn1(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)
        # flatten
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug1_fc_g2(x1)
        x1 = self.dropout(x1)

        # deal drug2
        x2 = self.node_embedder(x2)
        x2 = self.drug1_attn1(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.drug1_conv1(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.drug1_conv2(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.drug1_conv3(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2)
        # flatten
        x2 = self.relu(self.drug1_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.drug1_fc_g2(x2)
        x2 = self.dropout(x2)

        # deal cell
        cline_gene = cline_gene.unsqueeze(-1)
        cline_mutation = cline_mutation.unsqueeze(-1)
        cline_dependency = cline_dependency.unsqueeze(-1)
        cline_copynumber = cline_copynumber.unsqueeze(-1)
        x_cell = torch.cat((cline_gene, cline_mutation, cline_dependency, cline_copynumber), dim=-1)

        x_cell = x_cell.type(torch.float32)
        cell1 = x_cell.view(batch_size, self.genes_nums, -1)
        # cellB = cellA.clone()
        cell1 = self.cell_projector(cell1).float()
        cell_mask = self.causal_mask_layer(cell1)
        cell1_c = cell_mask[:, :, 0].unsqueeze(-1) * cell1
        cell1_t = cell_mask[:, :, 1].unsqueeze(-1) * cell1
        cell1_c = cell1_c.reshape(cell1_c.size(0), -1)
        cell1_t = cell1_t.reshape(cell1_t.size(0), -1)

        cell1_c = cell1_c.reshape(cell1_c.size(0), -1)
        cell1_t = cell1_c.reshape(cell1_t.size(0), -1)
        if self.args.get_cell_embedding:
            cell_embedding = cell1_c
            return cell_embedding

        # concat
        combined_c = torch.cat([x1, x2, cell1_c], dim=1)
        combined_t = torch.cat([x1, x2, cell1_t], dim=1)

        output_c = self.mlp_c(combined_c)
        output_t = self.mlp_t(combined_t)
        return output_c, output_t
