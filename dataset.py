import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from tqdm import tqdm


class TestbedDataset(InMemoryDataset):
    def __init__(self, root=None, dataset=None,ESM=None,Ligand_graph=None,smiles=None,y=None,ESM_global_dic=None,GIN_dic = None,
                  transform=None,protein_edge_index=None,
                 pre_transform=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(ESM_global_dic,ESM, Ligand_graph,smiles,GIN_dic,y,protein_edge_index)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass


    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):

        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def process(self,ESM_global_dic, ESM, Ligand_graph,smiles,GIN_dic,y,protein_edge_index):

        data_list = []

        # for name in smile_graph:
        for name in tqdm(Ligand_graph, desc="Processing"):
            x,edge_index,edge_attr,atom_size = Ligand_graph[name]
            ESM_global = ESM_global_dic[name]
            ESM_emb = ESM[name]
            smiles_seq = smiles[name]
            GIN_emb = GIN_dic[name]
            labels = y[name]
            pro_edge_index = protein_edge_index[name].to('cpu')


            GCNData = DATA.Data(
                x=torch.Tensor(x),edge_attr=torch.LongTensor(edge_attr),
                edge_index=torch.LongTensor(edge_index),
                y=torch.FloatTensor([labels]),SMILES=torch.LongTensor([smiles_seq]),pro_edge_index=torch.LongTensor(pro_edge_index)
                                )
            GCNData.__setitem__('atom_size', torch.LongTensor([atom_size]))
            GCNData.ESM = torch.FloatTensor(ESM_emb)
            GCNData.ESM_global = torch.FloatTensor(ESM_global).unsqueeze(0)
            GCNData.GIN_emb = torch.FloatTensor(GIN_emb).unsqueeze(0)
            GCNData.pdb_name = name
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        # print(data_list)
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

