import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset
# from torch_geometric.data import Dataset
import torch
from ogb.graphproppred import PygGraphPropPredDataset

from utils import getJointFingerprints


class JointFingerprintsDataset(Dataset):
    def __init__(self):
        df_smi = pd.read_csv(f"dataset/ogbg-molhiv/mapping/mol.csv.gz".replace("-", "_"))
        smiles = df_smi["smiles"]
        mols = [Chem.MolFromSmiles(s) for s in smiles.values]
        fingerprints = [torch.tensor(getJointFingerprints(m)) for m in mols]
        self.data = fingerprints

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SMILESDataset(Dataset):
    def __init__(self):
        df_smi = pd.read_csv(f"dataset/ogbg-molhiv/mapping/mol.csv.gz".replace("-", "_"))
        smiles = df_smi["smiles"]
        self.data = smiles.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FusionDataset(Dataset):
    def __init__(self, mode='train',
                 embeddings_path='./dataset/embeddings_fusion/GATModel3Pooled/experiment0SAG/fused_embeddings.pt',
                 labels_path='./dataset/embeddings_fusion/GATModel3Pooled/experiment0SAG/labels.pt'):
        self.data = torch.load(embeddings_path)
        self.labels = torch.load(labels_path)

        ogb_dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='dataset/')
        split = ogb_dataset.get_idx_split()
        if mode == 'train':
            self.indices = [int(split['train'][idx]) for idx in range(len(split['train']))]
        elif mode == 'valid':
            self.indices = [int(split['valid'][idx]) for idx in range(len(split['valid']))]
        elif mode == 'test':
            self.indices = [int(split['test'][idx]) for idx in range(len(split['test']))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.data[index].detach(), self.labels[index]
