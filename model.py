import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import softmax, to_dense_batch
import math
from torch.nn import LeakyReLU
from torch_geometric.nn import global_mean_pool,GCNConv
from Attention import MultiheadAttention

class MMPDIM_DTA(nn.Module):
    def __init__(self, ):
        super(MMPDIM_DTA, self).__init__()
        dropout = 0.3
        self.protein_module = Protein_module(input_dim=128, hidden_dim=128,output_dim=128)
        self.Gradformer = Gradformer(num_layers=2,num_heads=4,pe_origin_dim=20,pe_dim=36,hidden_dim=128,dropout=dropout)
        self.esm_lin = nn.Linear(1280,128)
        self.emb = nn.Embedding(65, 128)
        self.self_att = EncoderLayer(128, 128, dropout, dropout, 2)

        self.smi_module = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 0),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, 1, 0),
            nn.ReLU()
        )
        self.esm_norm = nn.LayerNorm(1280)

        self.gin_norm = nn.LayerNorm(300)
        self.MLP = nn.Sequential(
            nn.Linear(1280 + 128+128+300, 1024),nn.Dropout(dropout),nn.ReLU(),
            nn.Linear(1024, 512),nn.Dropout(dropout),nn.ReLU(),nn.Linear(512, 1))
    def forward(self, data):

        x,  edge_attr,  edge_index, ESM_global, ESM,    ESM_batch , smiles,gin, batch,  sph,    pe, pro_edge_index\
            = data.x,data.edge_attr,data.edge_index,data.ESM_global,data.ESM,data.ESM_batch,\
            data.SMILES,data.GIN_emb,data.batch,data.sph,data.pe,data.pro_edge_index
        ESM = self.esm_lin(ESM)
        ESM = self.protein_module(ESM,pro_edge_index,ESM_batch)



        smiles = self.emb(smiles)
        smiles,_ = self.self_att(smiles,smiles)
        smiles = self.smi_module(smiles.permute(0, 2, 1)).mean(dim=2)

        sph = process_hop(sph.float(), gamma=0.6, hop=2, slope=0.1)
        x,att = self.Gradformer(x, pe, edge_index, edge_attr, sph,ESM,batch)

        esm_global  =   self.esm_norm(ESM_global)
        gin         =   self.gin_norm(gin)
        #x = self.graph_norm(x)
        x = torch.cat([esm_global,x, smiles,gin], dim=1)
        x = self.MLP(x)
        return x



class Protein_module(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x,mask = to_dense_batch(x,batch)
        x = x[:, :2111, :]
        x = F.pad(x, (0, 0, 0, 2111 - x.shape[1]))
        return x
class Gradformer(torch.nn.Module):
    def __init__(self, num_layers,num_heads,pe_origin_dim,pe_dim,hidden_dim,dropout,node_dim=44):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe_norm = nn.BatchNorm1d(pe_origin_dim)
        self.node_lin = nn.Linear(node_dim, hidden_dim-pe_dim)
        self.edge_lin = nn.Linear(10, hidden_dim)
        self.pe_lin = nn.Linear(pe_origin_dim, pe_dim)
        self.gconvs = nn.ModuleList()
        self.middle_layers_1 = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(EELA(hidden_dim=hidden_dim, num_heads=num_heads, local_attn_dropout_ratio=self.dropout, local_ffn_dropout_ratio=self.dropout))
            self.middle_layers_1.append(nn.BatchNorm1d(hidden_dim))
            self.attentions.append(MultiheadAttention(2,hidden_dim,dropout=self.dropout))
        self.cross_attention = EncoderLayer(hidden_dim, 256, dropout, dropout, 2)
    def forward(self, x, pe, edge_index, edge_attr, sph,ESM,batch):
        pe = self.pe_norm(pe)
        x = torch.cat((self.node_lin(x), self.pe_lin(pe)), 1)
        edge_attr = self.edge_lin(edge_attr.float())

        x, mask = to_dense_batch(x, batch)
        x,att = self.cross_attention(x, ESM)
        x = x[mask]

        for i in range(self.num_layers):
            x = self.gconvs[i](x, edge_index, edge_attr)
            x = F.dropout(x,p=self.dropout,training=self.training)
            x = self.middle_layers_1[i](x)
            x,mask = to_dense_batch(x,batch)
            x = self.attentions[i](x,x,x,sph,~mask)
            x = x[mask]
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x,batch)
        return x,att

def process_hop(sph, gamma=0.6, hop=2, slope=0.1):
    # print(sph)
    leakyReLU = LeakyReLU(negative_slope=slope)
    sph = sph.unsqueeze(1)
    sph = sph - hop
    sph = leakyReLU(sph)
    sp = torch.pow(gamma, sph)
    return sp
class EELA(torch_geometric.nn.MessagePassing):  # ogbg-molpcba
    def __init__(self, hidden_dim: int, num_heads: int,
                 local_attn_dropout_ratio: float = 0.0,
                 local_ffn_dropout_ratio: float = 0.0):

        super().__init__(aggr='add', node_dim=0)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.local_attn_dropout_ratio = local_attn_dropout_ratio

        self.linear_dst = nn.Linear(hidden_dim, hidden_dim)
        self.linear_src_edge = nn.Linear(2 * hidden_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(local_ffn_dropout_ratio),
        )

    def reset_parameters(self):
        self.linear_dst.reset_parameters()
        self.linear_src_edge.reset_parameters()
        for layer in self.ffn:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        local_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        local_out = local_out.view(-1, self.hidden_dim)
        x = self.ffn(local_out)

        return x

    def message(self, x_i, x_j, edge_attr, edge_index_i, size_i):
        H, C = self.num_heads, self.hidden_dim // self.num_heads

        x_dst = self.linear_dst(x_i).view(-1, H, C)
        m_src = self.linear_src_edge(torch.cat([x_j, edge_attr], dim=-1)).view(-1, H, C)

        alpha = (x_dst * m_src).sum(dim=-1) / math.sqrt(C)

        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.local_attn_dropout_ratio, training=self.training)

        return m_src * alpha.unsqueeze(-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        att = x

        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x,att

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        #        self.gelu = GELU()
        self.gelu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None):
        y = self.self_attention_norm(x)
        kv = self.self_attention_norm(kv)
        y,att = self.self_attention(y, kv, kv, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x,att