import torch
from torch import nn
import torch.nn.functional as F
import math
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


# DrugA-drugB mutual-attention encoder, only output drugA embedding
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


# CellA-cellB mutual-attention encoder, only output cellA embedding
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


class synergyx(BaseModel):
    def __init__(self, gene_dim, dropout=0.5,
                 num_attention_heads=8,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 max_length=50,
                 input_dim_drug=2586,
                 output_dim=2560,
                 **args):

        super(synergyx, self).__init__(**args)
        self.in_channel = 4
        self.max_length = 50

        self.genes_nums = 14749

        self.patch = 150

        hidden_size = 128
        self.cell_conv = CellCNN(in_channel=self.in_channel, feat_dim=hidden_size)

        intermediate_size = hidden_size * 2
        self.drug_emb = Embeddings(input_dim_drug, hidden_size, self.patch, hidden_dropout_prob)
        self.drug_SA = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                               hidden_dropout_prob)
        self.cell_SA = EncoderCell(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                   hidden_dropout_prob)
        self.drug_CA = EncoderCA(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                 hidden_dropout_prob)
        self.cell_CA = EncoderCellCA(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                     hidden_dropout_prob)
        self.drug_cell_CA = EncoderD2C(hidden_size, intermediate_size, num_attention_heads,
                                       attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_SSA = EncoderSSA(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                   hidden_dropout_prob)
        self.cell_SSA = EncoderCellSSA(hidden_size, intermediate_size, num_attention_heads,
                                       attention_probs_dropout_prob, hidden_dropout_prob)

        self.head = FusionHead()

        self.cell_fc = nn.Sequential(
            nn.Linear(self.patch * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.drug_fc = nn.Sequential(
            nn.Linear(self.patch * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
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
        cellA = x_cell.view(batch_size, self.genes_nums, -1)
        cellB = cellA.clone()

        cellA = self.cell_conv(cellA)
        cellB = self.cell_conv(cellB)

        if self.args.get_cell_embedding:
            cell_embedding = cellA.reshape(cellA.size(0), -1)
            return cell_embedding

        # layer1
        cellA0 = cellA
        cellA, drugA, attn9, attn10 = self.drug_cell_CA(cellA, drugA, drugA_attention_mask, None)
        cellB, drugB, attn11, attn12 = self.drug_cell_CA(cellB, drugB, drugB_attention_mask, None)
        cellA1 = cellA
        cellB1 = cellB
        # layer2
        drugA, attn5 = self.drug_SA(drugA, drugA_attention_mask)
        drugB, attn6 = self.drug_SA(drugB, drugB_attention_mask)
        cellA, attn7 = self.cell_SA(cellA, None)
        cellB, attn8 = self.cell_SA(cellB, None)
        cellA2 = cellA
        cellB2 = cellB
        # layer3
        drugA, drugB, attn1, attn2 = self.drug_CA(drugA, drugB, drugA_attention_mask, drugB_attention_mask)
        cellA, cellB, attn3, attn4 = self.cell_CA(cellA, cellB, None, None)
        cellA3 = cellA
        cellB3 = cellB

        drugA_embed = self.drug_fc(drugA.view(-1, drugA.shape[1] * drugA.shape[2]))
        drugB_embed = self.drug_fc(drugB.view(-1, drugB.shape[1] * drugB.shape[2]))
        cellA_embed = self.cell_fc(cellA.contiguous().view(-1, cellA.shape[1] * cellA.shape[2]))
        cellB_embed = self.cell_fc(cellB.contiguous().view(-1, drugA.shape[1] * cellB.shape[2]))

        cell_embed = torch.cat((cellA_embed, cellB_embed), 1)
        drug_embed = torch.cat((drugA_embed, drugB_embed), 1)
        output = self.head(cell_embed, drug_embed)

        return output

    def init_weights(self):

        self.head.init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

        self.cell_conv.init_weights()