import esm
import torch_geometric
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos
import atom3d.util.formats as fo
import torch
import torch_cluster
from rdkit import Chem
import torch_geometric.transforms as T
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import softmax, to_dense_batch
import math
from torch.nn import LeakyReLU
from torch_geometric.nn import global_mean_pool,GCNConv
import torch
import math, copy
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
def graph_construction_and_featurization(ligand_file):
    graphs = []
    success = []
    # smi = smiles
    mol = Chem.MolFromMol2File(ligand_file)
    g = mol_to_bigraph(mol, add_self_loop=True,
                       node_featurizer=PretrainAtomFeaturizer(),
                       edge_featurizer=PretrainBondFeaturizer(),
                       canonical_atom_order=False)
    graphs.append(g)
    return graphs, success

def clones(module, N):
    """Product N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, sph, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    scores = scores * sph
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'),)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiheadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, sph, mask=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        # (batch_size, seq_length, d_model)

        # 1) Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for x in (query, key, value)]
        # (batch_size, h, seq_length, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, sph, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
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
        sph = torch.unsqueeze(sph, 0)
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
CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64,"~":65}
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(17),degree(7),formal charge(1),
    radical electrons(1),hybridization(6),aromatic(1),hydrogen atoms attached(5),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  # 17-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]  # 17+7+2+6+1=33

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])  # 33+5=38
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3 =41
    return results
def bond_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)
def mol2_to_graph(mol2):
    mol = Chem.MolFromMol2File(mol2)
    smiles = Chem.MolToSmiles(mol)
    smiles = label_smiles(smiles)
    c_size = mol.GetNumAtoms()

    atom_feats = np.array([atom_features(a, explicit_H=False) for a in mol.GetAtoms()])

    chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                              useLegacyImplementation=False)
    chiral_arr = np.zeros([c_size, 3])
    for (i, rs) in chiralcenters:
        if rs == 'R':
            chiral_arr[i, 0] = 1
        elif rs == 'S':
            chiral_arr[i, 1] = 1
        else:
            chiral_arr[i, 2] = 1
    atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)


    num_bonds = mol.GetNumBonds()
    # print(num_bonds)
    edge_attr = []
    src_list = []
    dst_list = []
    edge_index=[]
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = bond_features(bond, use_chirality=True)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        edge_attr.append(bond_feats)
        edge_attr.append(bond_feats)
    edge_index.append(src_list)
    edge_index.append(dst_list)
    return atom_feats,edge_index,edge_attr,torch.LongTensor([smiles]),c_size
def label_smiles(line):
    X = np.zeros(100)
    for i, ch in enumerate(line[:100]):
        X[i] = CHAR_SMI_SET[ch]
    return X.astype(int)
_amino_acids = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H','ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q','ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
standard_amino_acids = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS","ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP","TYR", "VAL"}
def load_protein_graph(protein_path):
    # print(protein_path)
    protein_df = fo.bp_to_df(fo.read_pdb(protein_path))
    protein = protein_df
    N_coords = protein[protein.name == 'N'][['x', 'y', 'z']].to_numpy()
    CA_coords = protein[protein.name == 'CA'][['x', 'y', 'z']].to_numpy()
    C_coords = protein[protein.name == 'C'][['x', 'y', 'z']].to_numpy()
    O_coords = protein[protein.name == 'O'][['x', 'y', 'z']].to_numpy()
    max_ = min([CA_coords.shape[0], C_coords.shape[0], N_coords.shape[0], O_coords.shape[0]])
    N_coords = N_coords[:max_, :]
    CA_coords = CA_coords[:max_, :]
    C_coords = C_coords[:max_, :]
    O_coords = O_coords[:max_, :]
    coords = np.stack((N_coords, CA_coords, C_coords, O_coords), axis=1)
    with torch.no_grad():
        coords = torch.from_numpy(coords).float()
        max_ = min([CA_coords.shape[0], C_coords.shape[0], N_coords.shape[0], O_coords.shape[0]])
        valid_resnames = protein[protein.name == 'CA']['resname'][:max_]
        filtered_resnames = [a for a in valid_resnames if a in standard_amino_acids]
        seq = [_amino_acids[a] for a in filtered_resnames]
        seq_str = "".join(seq)
        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf
        X_ca = coords[:, 1]
        edge_index = torch_cluster.radius_graph(X_ca, r=5.0)
        return seq_str, edge_index


def inference(protein_pdb_file,ligand_file,model_path,MMPDIM_DTA):
    dataset, success = graph_construction_and_featurization(ligand_file)
    mol_emb = []
    bg = dataset[0]
    bg = bg.to('cpu')
    nfeats = [bg.ndata.pop('atomic_number').to('cpu'),
              bg.ndata.pop('chirality_type').to('cpu')]
    efeats = [bg.edata.pop('bond_type').to('cpu'),
              bg.edata.pop('bond_direction_type').to('cpu')]
    model = load_pretrained('gin_supervised_contextpred').to('cpu')
    model.eval()
    readout = AvgPooling()
    with torch.no_grad():
        node_repr = model(bg, nfeats, efeats)
    mol_emb.append(readout(bg, node_repr))
    GIN_emb = np.array(readout(bg, node_repr))

    seq,protein_edge_index = load_protein_graph(protein_pdb_file)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    model.to('cuda')
    with torch.no_grad():
        try:
            batch_tokens = batch_tokens.to('cuda')
            results = model(batch_tokens, repr_layers=[33])
        except:
            model.to('cpu')
            batch_tokens = batch_tokens.to('cpu')
            results = model(batch_tokens, repr_layers=[33])
    token_representations = results["representations"][33][0][1: -1].to('cpu')

    protein_x = torch.FloatTensor(token_representations)
    protein_global =  torch.FloatTensor(token_representations.mean(dim=0)).unsqueeze(0)
    x,edge_index,edge_attr,smiles,atom_size = mol2_to_graph(ligand_file)
    edge_index = np.array(edge_index)
    N = len(x)
    adj = torch.zeros([N, N])
    adj[edge_index[0, :], edge_index[1, :]] = True
    sph, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))

    device = torch.device('cuda')
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    data = torch_geometric.data.Data(x=torch.Tensor(x).to(device),edge_attr=torch.LongTensor(edge_attr).to(device),edge_index=torch.LongTensor(edge_index).to(device))
    data = transform(data)
    data.pro_edge_index = protein_edge_index.to(device)
    data.sph = sph.to(device)
    data.SMILES = smiles.to(device)
    data.ESM = protein_x.to(device)
    data.ESM_global = protein_global.to(device)
    data.GIN_emb = torch.FloatTensor(GIN_emb).to(device)
    data.ESM_batch = torch.zeros(len(protein_x), dtype=torch.long).to(device)
    data.x_batch = torch.zeros(data.num_nodes, dtype=torch.long).to(device)
    MMPDIM_DTA = MMPDIM_DTA().to(device)
    MMPDIM_DTA.load_state_dict(torch.load(model_path))
    MMPDIM_DTA.eval()
    with torch.no_grad():
        output = MMPDIM_DTA(data)
    return output

if __name__ == '__main__':
    import sys

    # protein_pdb_file = '1nvq_protein.pdb'
    # ligand_file = '1nvq_ligand.mol2'
    # model_path = 'best_model_1.pt'
    model_path = sys.argv[1]
    protein_pdb_file = sys.argv[2]
    drug_file = sys.argv[3]
    affinity = inference(protein_pdb_file,drug_file,model_path,MMPDIM_DTA)
    print('Affinity=',affinity.item())













