import os
import pandas as pd
from dataset import TestbedDataset
import torch_geometric.transforms as T
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos
from tqdm import tqdm
from rdkit import Chem
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
import networkx as nx
import torch_cluster
import atom3d.util.formats as fo
import esm
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
    return atom_feats,edge_index,edge_attr,smiles,c_size
def label_smiles(line):
    X = np.zeros(100)
    for i, ch in enumerate(line[:100]):
        X[i] = CHAR_SMI_SET[ch]
    return X

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
def GIN_load(ligand_file,GIN_model):
    dataset, success = graph_construction_and_featurization(ligand_file)
    mol_emb = []
    bg = dataset[0]
    bg = bg.to('cpu')
    nfeats = [bg.ndata.pop('atomic_number').to('cpu'),
              bg.ndata.pop('chirality_type').to('cpu')]
    efeats = [bg.edata.pop('bond_type').to('cpu'),
              bg.edata.pop('bond_direction_type').to('cpu')]

    model = GIN_model
    readout = AvgPooling()
    with torch.no_grad():
        node_repr = model(bg, nfeats, efeats)
    mol_emb.append(readout(bg, node_repr))
    GIN_emb = np.array(readout(bg, node_repr))
    return GIN_emb
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
def load_pdb(file,esm_model,alphabet):
    seq, protein_edge_index = load_protein_graph(file)


    data = [("protein1", seq)]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    model = esm_model
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
    protein_global = torch.FloatTensor(token_representations.mean(dim=0)).unsqueeze(0)
    return protein_x,protein_global,protein_edge_index

def compute_sph(edge_index):
    edge_index = np.array(edge_index)
    N = len(x)
    adj = torch.zeros([N, N])
    adj[edge_index[0, :], edge_index[1, :]] = True
    sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
    return sp
if __name__ == '__main__':

    data_select = 'test2013'
    file_path = '/PDBbind2020/'+data_select+'/protein'
    file_list = os.listdir(file_path) # we provide test2016 and test2013 on Google Drive (test2016.tar.xz and test2013.tar.xz)
    # load pre-trained gin
    GIN_model = load_pretrained('gin_supervised_contextpred').to('cpu')
    GIN_model.eval()
    # load pre-trained esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model.eval()
    esm_model.to('cuda')

    ESM_x = {} # protein graph node feature vector
    ESM_global={} # protein global feature
    protein_edge_index_={}  # protein edge index
    Ligand_graph_dic= {} # ligand graph
    smiles_dic = {} # ligand smiles string
    sph_dic = {}   # decay mask
    GIN_dic = {}  # drug gin global feature

    # load_affinity_file
    affinity = {}
    affinity_df = pd.read_csv('affinity_data.csv') # we provide it on  Google Drive (affinity_data.csv)
    for _, row in affinity_df.iterrows():
        affinity[row[0]] = row[1]
    n=0
    for name in tqdm(file_list):
        name = name.split('_')[0]
        protein_file = '/PDBbind2020/'+data_select+'/protein'+'/'+name+'_protein.pdb'
        ligand_file = '/PDBbind2020/'+data_select+'/mol2'+'/'+name+'_ligand.mol2'
        protein_x,protein_global,protein_edge_index=load_pdb(protein_file,esm_model,alphabet)
        x,edge_index,edge_attr,smiles,atom_size = mol2_to_graph(ligand_file)



        GIN_dic[name] = GIN_load(ligand_file,GIN_model)
        ESM_global[name]=protein_global
        ESM_x[name]=protein_x
        protein_edge_index_[name]=protein_edge_index
        Ligand_graph_dic[name] = (x, edge_index, edge_attr, atom_size)



        sph_dic[name] = compute_sph(edge_index)

    # print(len(sph_dic))

        smiles_dic[name]=smiles

    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    test_data = TestbedDataset(root='data', dataset=data_select,ESM_global_dic=ESM_global,ESM = ESM_x,Ligand_graph = Ligand_graph_dic,smiles=smiles_dic,GIN_dic =GIN_dic ,y=affinity,protein_edge_index=protein_edge_index_,pre_transform=transform)

    # save sph file
    # put it in MMPDIM-DTA/data/processed
    # we provide it on  Google Drive (sph_zong.pt)  https://drive.google.com/drive/folders/1SyVzxgTGPr9dtBRbexzlLA5PMmUuJKPl?hl=zh-cn
    torch.save(sph_dic,data_select + 'sph.pt')