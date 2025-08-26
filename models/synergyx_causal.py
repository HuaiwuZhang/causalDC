import torch
from torch import nn
import torch.nn.functional as F
import math
from src.base.model import BaseModel
import shap
import numpy as np
from captum.attr import IntegratedGradients, GradientShap


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        # Normalize input_tensor
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # Apply scaling and bias
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
        # input_ids = input_ids.unsqueeze(0)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs_0

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, drugA, drugB, drugA_attention_mask):
        # update drugA
        mixed_query_layer = self.query(drugA)
        mixed_key_layer = self.key(drugB)
        mixed_value_layer = self.value(drugB)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if drugA_attention_mask == None:
            attention_scores = attention_scores
        else:
            attention_scores = attention_scores + drugA_attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs_0

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs_0 = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs_0

class Attention_SSA(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention_SSA, self).__init__()
        self.self = CrossAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, drugA, drugB, drugA_attention_mask):
        drugA_self_output, attention_probs_0 = self.self(drugA, drugB, drugA_attention_mask)
        drugA_attention_output = self.output(drugA_self_output, drugA)
        return drugA_attention_output, attention_probs_0

class Attention_CA(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention_CA, self).__init__()
        self.self = CrossAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, drugA, drugB, drugA_attention_mask, drugB_attention_mask):
        drugA_self_output, drugA_attention_probs_0 = self.self(drugA, drugB, drugA_attention_mask)
        drugB_self_output, drugB_attention_probs_0 = self.self(drugB, drugA, drugB_attention_mask)
        drugA_attention_output = self.output(drugA_self_output, drugA)
        drugB_attention_output = self.output(drugB_self_output, drugB)
        return drugA_attention_output, drugB_attention_output, drugA_attention_probs_0, drugB_attention_probs_0

class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs_0 = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs_0

    # Cell self-attention encoder

class EncoderCell(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderCell, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size))

    def forward(self, hidden_states, attention_mask):
        hidden_states_1 = self.LayerNorm(hidden_states)
        attention_output, attention_probs_0 = self.attention(hidden_states_1, attention_mask)
        hidden_states_2 = hidden_states_1 + attention_output
        hidden_states_3 = self.LayerNorm(hidden_states_2)

        hidden_states_4 = self.dense(hidden_states_3)

        layer_output = hidden_states_2 + hidden_states_4

        return layer_output, attention_probs_0

    # Drug-drug mutual-attention encoder

class EncoderCA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderCA, self).__init__()
        self.attention_CA = Attention_CA(hidden_size, num_attention_heads,
                                         attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, drugA, drugB, drugA_attention_mask, drugB_attention_mask):
        drugA_attention_output, drugB_attention_output, drugA_attention_probs_0, drugB_attention_probs_0 = self.attention_CA(
            drugA, drugB, drugA_attention_mask, drugB_attention_mask)
        drugA_intermediate_output = self.intermediate(drugA_attention_output)
        drugA_layer_output = self.output(drugA_intermediate_output, drugA_attention_output)
        drugB_intermediate_output = self.intermediate(drugB_attention_output)
        drugB_layer_output = self.output(drugB_intermediate_output, drugB_attention_output)
        return drugA_layer_output, drugB_layer_output, drugA_attention_probs_0, drugB_attention_probs_0

    # Cell-cell mutual-attention encoder

class EncoderCellCA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderCellCA, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention_CA = Attention_CA(hidden_size, num_attention_heads,
                                         attention_probs_dropout_prob, hidden_dropout_prob)

        self.dense = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size))

    def forward(self, cellA, cellB, cellA_attention_mask=None, cellB_attention_mask=None):
        cellA_1 = self.LayerNorm(cellA)
        cellB_1 = self.LayerNorm(cellB)

        cellA_attention_output, cellB_attention_output, cellA_attention_probs_0, cellB_attention_probs_0 = self.attention_CA(
            cellA, cellB, cellA_attention_mask, cellB_attention_mask)

        # cellA_output
        cellA_2 = cellA_1 + cellA_attention_output
        cellA_3 = self.LayerNorm(cellA_2)
        cellA_4 = self.dense(cellA_3)
        cellA_layer_output = cellA_2 + cellA_4

        # cellB_output
        cellB_2 = cellB_1 + cellB_attention_output
        cellB_3 = self.LayerNorm(cellB_2)
        cellB_4 = self.dense(cellB_3)
        cellB_layer_output = cellB_2 + cellB_4

        return cellA_layer_output, cellB_layer_output, cellA_attention_probs_0, cellB_attention_probs_0

# Drug-cell mutual-attention encoder
class EncoderD2C(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderD2C, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention_CA = Attention_CA(hidden_size, num_attention_heads,
                                         attention_probs_dropout_prob, hidden_dropout_prob)

        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size))

    def forward(self, cell, drug, drug_attention_mask, cell_attention_mask=None):
        cell_1 = self.LayerNorm(cell)
        cell_attention_output, drug_attention_output, cell_attention_probs_0, drug_attention_probs_0 = self.attention_CA(
            cell_1, drug, cell_attention_mask, drug_attention_mask)
        # cell_output
        cell_2 = cell_1 + cell_attention_output
        cell_3 = self.LayerNorm(cell_2)
        cell_4 = self.dense(cell_3)
        cell_layer_output = cell_2 + cell_4
        # drug_output
        drug_intermediate_output = self.intermediate(drug_attention_output)
        drug_layer_output = self.output(drug_intermediate_output, drug_attention_output)

        return cell_layer_output, drug_layer_output, cell_attention_probs_0, drug_attention_probs_0

class EncoderSSA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderSSA, self).__init__()
        self.attention_SSA = Attention_SSA(hidden_size, num_attention_heads,
                                           attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, drugA, drugB, drugA_attention_mask):
        drugA_attention_output, drugA_attention_probs_0 = self.attention_SSA(drugA, drugB, drugA_attention_mask)
        drugA_intermediate_output = self.intermediate(drugA_attention_output)
        drugA_layer_output = self.output(drugA_intermediate_output, drugA_attention_output)
        return drugA_layer_output, drugA_attention_probs_0

class EncoderCellSSA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderCellSSA, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention_SSA = Attention_SSA(hidden_size, num_attention_heads,
                                           attention_probs_dropout_prob, hidden_dropout_prob)

        self.dense = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size))

    def forward(self, cellA, cellB, cellA_attention_mask=None):
        cellA_1 = self.LayerNorm(cellA)
        cellB_1 = self.LayerNorm(cellB)

        cellA_attention_output, cellA_attention_probs_0 = self.attention_SSA(cellA, cellB, cellA_attention_mask)

        # cellA_output
        cellA_2 = cellA_1 + cellA_attention_output
        cellA_3 = self.LayerNorm(cellA_2)
        cellA_4 = self.dense(cellA_3)
        cellA_layer_output = cellA_2 + cellA_4

        return cellA_layer_output, cellA_attention_probs_0

class FusionHead(torch.nn.Module):
    def __init__(self, args=None, out_channels=256):
        super(FusionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 40, int(out_channels * 20)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(out_channels * 20), out_channels * 10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(out_channels * 10, 1),
        )
        self.tanh = nn.Tanh()
        self.args = args

    def forward(self, x_cell_embed, drug_embed):
        out = torch.cat((x_cell_embed, drug_embed), dim=1)
        out = self.fc(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

class CellCNN(nn.Module):
    def __init__(self, in_channel=3, feat_dim=None, args=None):
        super(CellCNN, self).__init__()

        max_pool_size = [2, 2, 3, 8]
        drop_rate = 0.2
        kernel_size = [16, 16, 16, 16]

        in_channels = [4, 16, 32, 64]
        out_channels = [16, 32, 64, 128]

        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
            nn.Conv1d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=kernel_size[3]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[3]),
        )

        self.cell_linear = nn.Linear(out_channels[3], feat_dim)

    def forward(self, x):

        x = x.transpose(1, 2)
        x_cell_embed = self.cell_conv(x)
        x_cell_embed = x_cell_embed.transpose(1, 2)
        x_cell_embed = self.cell_linear(x_cell_embed)

        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)

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

    def get_combined_weight(self):
        combined_weight = self.proj_layers[0].weight.data
        for layer in self.proj_layers[1:]:
            combined_weight = layer.weight.data @ combined_weight
        return combined_weight

class CausalMask(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mask_generator = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        mask_logits = self.mask_generator(x)
        return torch.softmax(mask_logits, dim=-1)

class synergyx_causal(BaseModel):
    def __init__(self, gene_dim, dropout=0.5,
                 num_attention_heads=8,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 max_length=50,
                 input_dim_drug=2586,
                 output_dim=2560,
                 **args):

        super(synergyx_causal, self).__init__(**args)
        self.in_channel = 4
        self.max_length = 50

        self.genes_nums = 14749

        self.patch = 150

        hidden_size = 128
        self.cell_conv = CellCNN(in_channel=self.in_channel, feat_dim=hidden_size)
        self.cell_projector = MultiLevelProjection()

        intermediate_size = hidden_size * 2
        self.drug_emb = Embeddings(input_dim_drug, hidden_size, self.patch, hidden_dropout_prob)
        self.drug_SA_c = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                 hidden_dropout_prob)
        self.cell_SA_c = EncoderCell(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                     hidden_dropout_prob)
        self.drug_CA_c = EncoderCA(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                   hidden_dropout_prob)
        self.cell_CA_c = EncoderCellCA(hidden_size, intermediate_size, num_attention_heads,
                                       attention_probs_dropout_prob,
                                       hidden_dropout_prob)
        self.drug_SA_t = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                 hidden_dropout_prob)
        self.cell_SA_t = EncoderCell(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                     hidden_dropout_prob)
        self.drug_CA_t = EncoderCA(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                   hidden_dropout_prob)
        self.cell_CA_t = EncoderCellCA(hidden_size, intermediate_size, num_attention_heads,
                                       attention_probs_dropout_prob,
                                       hidden_dropout_prob)

        self.cell_fc_c = nn.Sequential(
            nn.Linear(self.patch * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.drug_fc_c = nn.Sequential(
            nn.Linear(self.patch * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.cell_fc_t = nn.Sequential(
            nn.Linear(self.patch * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.drug_fc_t = nn.Sequential(
            nn.Linear(self.patch * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.drug_cell_CA_c = EncoderD2C(hidden_size, intermediate_size, num_attention_heads,
                                         attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_cell_CA_t = EncoderD2C(hidden_size, intermediate_size, num_attention_heads,
                                         attention_probs_dropout_prob, hidden_dropout_prob)

        self.head_c = FusionHead()
        self.head_t = FusionHead()


        self.causal_mask_layer = CausalMask()

    def forward(self, drugA, drugB, cline_gene, cline_mutation, cline_dependency, cline_copynumber, get_causal_genes=False):

        batch_size = drugA.size(0)
        device = self.args.device
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

        cell1 = self.cell_projector(cell1)
        cell_mask = self.causal_mask_layer(cell1)
        if get_causal_genes:

            combined_weight = self.cell_projector.get_combined_weight()
            expanded_mask = torch.einsum('ijn,jk->ikn', cell_mask, combined_weight)
            causal_gene_score = torch.softmax(expanded_mask, dim=-1)[:, :, 0]
            print(torch.max(causal_gene_score))

            return causal_gene_score

        if self.args.causal_ablation_ratio < 1 and self.args.causal_ablation_ratio > 0:

            combined_weight = self.cell_projector.get_combined_weight()
            expanded_mask = torch.einsum('ijn,jk->ikn', cell_mask, combined_weight)
            causal_gene_score = torch.softmax(expanded_mask, dim=-1)[:, :, 0]
            ablation_gene_num = int(self.args.causal_ablation_ratio * self.genes_nums)
            _, topk_indices = torch.topk(causal_gene_score, k=ablation_gene_num, dim=1)
            x_cell = self.ablate_genes_3d(x_cell, topk_indices)
            cell1 = x_cell.view(batch_size, self.genes_nums, -1)
            cell1 = self.cell_projector(cell1)

        elif self.args.causal_ablation_ratio < 0 and self.args.causal_ablation_ratio > -1:
            causal_mask = cell_mask[:, :, 0]
            combined_weight = self.cell_projector.get_combined_weight()
            expanded_mask = torch.einsum('ijn,jk->ikn', cell_mask, combined_weight)
            causal_gene_score = torch.softmax(expanded_mask, dim=-1)[:, :, 0]
            ablation_gene_num = int(abs(self.args.causal_ablation_ratio) * self.genes_nums)
            _, topk_indices = torch.topk(causal_gene_score, k=ablation_gene_num, dim=1, largest=False)
            x_cell = self.ablate_genes_3d(x_cell, topk_indices)
            cell1 = x_cell.view(batch_size, self.genes_nums, -1)
            cell1 = self.cell_projector(cell1)

        cellA_c = cell_mask[:, :, 0].unsqueeze(-1) * cell1
        cellA_t = cell_mask[:, :, 1].unsqueeze(-1) * cell1
        cellB_c = cellA_c.clone()
        cellB_t = cellA_t.clone()
        if self.args.get_cell_embedding:
            cell_embedding = cellA_c.reshape(cellA_c.size(0), -1)
            return cell_embedding


        if True:
            # causal
            cellA_c, drugA_c, attn9, attn10 = self.drug_cell_CA_c(cellA_c, drugA, drugA_attention_mask, None)
            cellB_c, drugB_c, attn11, attn12 = self.drug_cell_CA_c(cellB_c, drugB, drugB_attention_mask, None)

            # trivial
            cellA_t, drugA_t, attn9, attn10 = self.drug_cell_CA_t(cellA_t, drugA, drugA_attention_mask, None)
            cellB_t, drugB_t, attn11, attn12 = self.drug_cell_CA_t(cellB_t, drugB, drugB_attention_mask, None)
        else:
            drugA_c = drugA
            drugB_c = drugB
            drugA_t = drugA
            drugB_t = drugB

        # Fusion layer2 ##############################

        # causal
        drugA_c, attn5 = self.drug_SA_c(drugA_c, drugA_attention_mask)
        drugB_c, attn6 = self.drug_SA_c(drugB_c, drugB_attention_mask)
        cellA_c, attn7 = self.cell_SA_c(cellA_c, None)
        cellB_c, attn8 = self.cell_SA_c(cellB_c, None)

        # trivial
        drugA_t, _ = self.drug_SA_t(drugA_t, drugA_attention_mask)
        drugB_t, _ = self.drug_SA_t(drugB_t, drugB_attention_mask)
        cellA_t, _ = self.cell_SA_t(cellA_t, None)
        cellB_t, _ = self.cell_SA_t(cellB_t, None)

        # Fusion layer3 ##############################

        # causal
        if True:
            drugA_c, drugB_c, attn1, attn2 = self.drug_CA_c(drugA_c, drugB_c, drugA_attention_mask,
                                                            drugB_attention_mask)
            cellA_c, cellB_c, attn3, attn4 = self.cell_CA_c(cellA_c, cellB_c, None, None)

        drugA_embed_c = self.drug_fc_c(drugA_c.view(-1, drugA_c.shape[1] * drugA_c.shape[2]))
        drugB_embed_c = self.drug_fc_c(drugB_c.view(-1, drugB_c.shape[1] * drugB_c.shape[2]))
        cellA_embed_c = self.cell_fc_c(cellA_c.contiguous().view(-1, cellA_c.shape[1] * cellA_c.shape[2]))
        cellB_embed_c = self.cell_fc_c(cellB_c.contiguous().view(-1, cellB_c.shape[1] * cellB_c.shape[2]))

        cell_embed_c = torch.cat((cellA_embed_c, cellB_embed_c), 1)
        drug_embed_c = torch.cat((drugA_embed_c, drugB_embed_c), 1)
        output_c = self.head_c(cell_embed_c, drug_embed_c)

        # trivial
        if True:
            drugA_t, drugB_t, attn1, attn2 = self.drug_CA_t(drugA_t, drugB_t, drugA_attention_mask,
                                                            drugB_attention_mask)
            cellA_t, cellB_t, attn3, attn4 = self.cell_CA_t(cellA_t, cellB_t, None, None)

        drugA_embed_t = self.drug_fc_t(drugA_t.view(-1, drugA_t.shape[1] * drugA_t.shape[2]))
        drugB_embed_t = self.drug_fc_t(drugB_t.view(-1, drugB_t.shape[1] * drugB_t.shape[2]))
        cellA_embed_t = self.cell_fc_t(cellA_t.contiguous().view(-1, cellA_t.shape[1] * cellA_t.shape[2]))
        cellB_embed_t = self.cell_fc_t(cellB_t.contiguous().view(-1, cellB_t.shape[1] * cellB_t.shape[2]))

        cell_embed_t = torch.cat((cellA_embed_t, cellB_embed_t), 1)
        drug_embed_t = torch.cat((drugA_embed_t, drugB_embed_t), 1)
        output_t = self.head_t(cell_embed_t, drug_embed_t)


        return output_c, output_t

    def init_weights(self):

        self.head.init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

        self.cell_conv.init_weights()

    def ablate_genes_3d(self, x, topk_indices, ablation_mode='zero'):
        batch_size, num_genes, num_features = x.shape
        k = topk_indices.size(1)
        device = x.device
        batch_idx = torch.arange(batch_size, device=device)[:, None].repeat(1, k)
        key_features = x[batch_idx, topk_indices, :]
        if ablation_mode == 'zero':

            key_features = torch.zeros_like(key_features)
        elif ablation_mode == 'noise':

            noise = torch.randn_like(key_features) * 0.1
            key_features = noise
        elif ablation_mode == 'mean':

            non_key_mask = torch.ones((batch_size, num_genes), dtype=bool, device=device)
            for i in range(batch_size):
                non_key_mask[i, topk_indices[i]] = False

            mean_vals = torch.stack([x[i][non_key_mask[i]].mean(dim=0) for i in range(batch_size)])

            key_features = mean_vals[:, None, :].repeat(1, k, 1)

        x_ablated = x.clone()

        x_ablated[batch_idx, topk_indices, :] = key_features

        return x_ablated

    class ModelWrapper(nn.Module):


        def __init__(self, model, drugA, drugB):
            super().__init__()
            self.model = model
            self.drugA = drugA
            self.drugB = drugB
            self.model.eval()

        def forward(self, x):
            if not x.requires_grad:
                x = x.clone().requires_grad_(True)
            batch_size = x.shape[0]
            drugA_rep = self.drugA.repeat(batch_size, 1, 1)
            drugB_rep = self.drugB.repeat(batch_size, 1, 1)
            gene = x[..., 0]
            mutation = x[..., 1]
            dependency = x[..., 2]
            copynumber = x[..., 3]
            output_c, _ = self.model(
                drugA_rep, drugB_rep,
                gene, mutation, dependency, copynumber
            )
            return output_c

    def integrated_gradients(self, drugA, drugB, cline_gene, cline_mutation, cline_dependency, cline_copynumber, sample_index=0,
                             baselines=None, n_steps=50):

        cline_gene = cline_gene.unsqueeze(-1)
        cline_mutation = cline_mutation.unsqueeze(-1)
        cline_dependency = cline_dependency.unsqueeze(-1)
        cline_copynumber = cline_copynumber.unsqueeze(-1)
        x_cell = torch.cat((cline_gene, cline_mutation, cline_dependency, cline_copynumber), dim=-1)
        target_x = x_cell[sample_index].unsqueeze(0)
        drugA_fixed = drugA[sample_index].unsqueeze(0)
        drugB_fixed = drugB[sample_index].unsqueeze(0)
        if not target_x.requires_grad:
            target_x = target_x.clone().requires_grad_(True)
        wrapped_model = self.ModelWrapper(self, drugA_fixed, drugB_fixed).to(x_cell.device)
        if baselines is None:
            baselines = torch.zeros_like(target_x)
        if not baselines.requires_grad:
            baselines = baselines.clone().requires_grad_(True)
        ig = IntegratedGradients(wrapped_model)
        attributions, delta = ig.attribute(
            target_x,
            baselines=baselines,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        gene_attrib = attributions[0].detach().cpu().numpy()
        gene_importance = np.mean(np.abs(gene_attrib), axis=1)
        top_genes = np.argsort(gene_importance)[-100:][::-1]

        return {
            "attributions": gene_attrib,
            "gene_importance": gene_importance,
            "top_genes": top_genes,
            "delta": delta.item()
        }

    def gradient_shap(self, drugA, drugB, cline_gene, cline_mutation, cline_dependency, cline_copynumber, sample_index=0,
                      n_background=20, stdevs=0.09, n_samples=100):
        cline_gene = cline_gene.unsqueeze(-1)
        cline_mutation = cline_mutation.unsqueeze(-1)
        cline_dependency = cline_dependency.unsqueeze(-1)
        cline_copynumber = cline_copynumber.unsqueeze(-1)
        x_cell = torch.cat((cline_gene, cline_mutation, cline_dependency, cline_copynumber), dim=-1)
        target_x = x_cell[sample_index].unsqueeze(0)
        drugA_fixed = drugA[sample_index].unsqueeze(0)
        drugB_fixed = drugB[sample_index].unsqueeze(0)
        if not target_x.requires_grad:
            target_x = target_x.clone().requires_grad_(True)
        n_background = min(n_background, x_cell.shape[0])
        background_x = x_cell[:n_background].detach().clone()
        wrapped_model = self.ModelWrapper(self, drugA_fixed, drugB_fixed).to(x_cell.device)
        gs = GradientShap(wrapped_model)
        attributions = gs.attribute(
            target_x,
            baselines=background_x,
            stdevs=stdevs,
            n_samples=n_samples
        )
        gene_attrib = attributions[0].detach().cpu().numpy()
        gene_importance = np.mean(np.abs(gene_attrib), axis=1)
        top_genes = np.argsort(gene_importance)[-100:][::-1]

        return {
            "attributions": gene_attrib,
            "gene_importance": gene_importance,
            "top_genes": top_genes
        }

    class ModelWrapper(nn.Module):

        def __init__(self, model, drugA, drugB):
            super().__init__()
            self.model = model
            self.drugA = drugA
            self.drugB = drugB
            self.model.eval()

        def forward(self, x):
            if not x.requires_grad:
                x = x.clone().requires_grad_(True)
            batch_size = x.shape[0]
            drugA_rep = self.drugA.repeat(batch_size, 1, 1)
            drugB_rep = self.drugB.repeat(batch_size, 1, 1)
            gene = x[..., 0]
            mutation = x[..., 1]
            dependency = x[..., 2]
            copynumber = x[..., 3]

            output_c, _ = self.model(
                drugA_rep, drugB_rep,
                gene, mutation, dependency, copynumber
            )
            return output_c

    def integrated_gradients(self, drugA, drugB, cline_gene, cline_mutation, cline_dependency, cline_copynumber, sample_index=0,
                             baselines=None, n_steps=50):

        cline_gene = cline_gene.unsqueeze(-1)
        cline_mutation = cline_mutation.unsqueeze(-1)
        cline_dependency = cline_dependency.unsqueeze(-1)
        cline_copynumber = cline_copynumber.unsqueeze(-1)
        x_cell = torch.cat((cline_gene, cline_mutation, cline_dependency, cline_copynumber), dim=-1)
        target_x = x_cell[sample_index].unsqueeze(0)
        drugA_fixed = drugA[sample_index].unsqueeze(0)
        drugB_fixed = drugB[sample_index].unsqueeze(0)


        if not target_x.requires_grad:
            target_x = target_x.clone().requires_grad_(True)
        wrapped_model = self.ModelWrapper(self, drugA_fixed, drugB_fixed).to(x_cell.device)
        if baselines is None:
            baselines = torch.zeros_like(target_x)
        if not baselines.requires_grad:
            baselines = baselines.clone().requires_grad_(True)
        ig = IntegratedGradients(wrapped_model)
        attributions, delta = ig.attribute(
            target_x,
            baselines=baselines,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        gene_attrib = attributions[0].detach().cpu().numpy()
        gene_importance = np.mean(np.abs(gene_attrib), axis=1)
        top_genes = np.argsort(gene_importance)[-100:][::-1]

        return {
            "attributions": gene_attrib,
            "gene_importance": gene_importance,
            "top_genes": top_genes,
            "delta": delta.item()
        }

    def gradient_shap(self, drugA, drugB, cline_gene, cline_mutation, cline_dependency, cline_copynumber, sample_index=0,
                      n_background=20, stdevs=0.09, n_samples=100):

        cline_gene = cline_gene.unsqueeze(-1)
        cline_mutation = cline_mutation.unsqueeze(-1)
        cline_dependency = cline_dependency.unsqueeze(-1)
        cline_copynumber = cline_copynumber.unsqueeze(-1)
        x_cell = torch.cat((cline_gene, cline_mutation, cline_dependency, cline_copynumber), dim=-1)
        target_x = x_cell[sample_index].unsqueeze(0)
        drugA_fixed = drugA[sample_index].unsqueeze(0)
        drugB_fixed = drugB[sample_index].unsqueeze(0)
        if not target_x.requires_grad:
            target_x = target_x.clone().requires_grad_(True)
        n_background = min(n_background, x_cell.shape[0])
        background_x = x_cell[:n_background].detach().clone()
        wrapped_model = self.ModelWrapper(self, drugA_fixed, drugB_fixed).to(x_cell.device)
        gs = GradientShap(wrapped_model)
        attributions = gs.attribute(
            target_x,
            baselines=background_x,
            stdevs=stdevs,
            n_samples=n_samples
        )

        gene_attrib = attributions[0].detach().cpu().numpy()

        gene_importance = np.mean(np.abs(gene_attrib), axis=1)

        top_genes = np.argsort(gene_importance)[-100:][::-1]

        return {
            "attributions": gene_attrib,
            "gene_importance": gene_importance,
            "top_genes": top_genes
        }