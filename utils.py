import torch
from rdkit.Chem import AllChem
from tqdm import tqdm
from datasets import *
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from models import GATModel3Pooled, NNModel2SAG, GINEModel5Pooled

def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))


def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]


def getJointFingerprints(mol):
    return getmaccsfingerprint(mol) + getmorganfingerprint(mol)


def generate_fusion_embeddings(model, save_folder):
    # folder has to exist

    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='dataset/')
    fingerprints = torch.load('./dataset/fingerprints/fingerprints_list.pt')
    model.eval()

    atom_encoder = AtomEncoder(emb_dim=64)
    bond_encoder = BondEncoder(emb_dim=32)

    dataset = list(DataLoader(dataset, batch_size=1))

    fused_embeddings = []
    labels = []

    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        fingerprint = fingerprints[idx]
        embedding = model.calculate_embedding(atom_encoder(data.x), data.edge_index, bond_encoder(data.edge_attr),
                                              data.batch).squeeze()
        fused_embeddings.append(torch.cat((embedding, fingerprint), dim=0).detach())
        labels.append(data.y)

    torch.save(fused_embeddings, f'{save_folder}/fused_embeddings.pt')
    torch.save(labels, f'{save_folder}/labels.pt')


if __name__ == '__main__':

    # model = GINEModel5Pooled(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5, pooling_type='topk')
    # model.load_state_dict(torch.load('./experiments/GINEModel5Pooled/experiment1Topk/models/model_3.pt'))
    #
    # save_dir = 'dataset/embeddings_fusion/GINEModel5TopK/experiment1TopK'
    #
    # generate_fusion_embeddings(model, save_dir)

    #################################################
    # embeddings1 = torch.load('./dataset/embeddings_fusion/GATModel3Pooled/experiment0SAG/fused_embeddings.pt')
    # embeddings2 = torch.load('./dataset/embeddings_fusion/GINEModel5TopK/experiment1TopK/fused_embeddings.pt')
    # embeddings3 = torch.load('./dataset/embeddings_fusion/NNModel2SAG/experiment0/fused_embeddings.pt')
    # labels = torch.load('./dataset/embeddings_fusion/GATModel3Pooled/experiment0SAG/labels.pt')
    #
    # fused_embeddings = []
    #
    # for idx in range(len(labels)):
    #     fused_embeddings.append(torch.cat((embeddings1[idx], embeddings2[idx], embeddings3[idx]), dim=0).detach())
    #
    # torch.save(fused_embeddings, f'dataset/embeddings_fusion/AllFused/fused_embeddings.pt')
    # torch.save(labels, f'dataset/embeddings_fusion/AllFused/labels.pt')
    ##################################################
    model1 = GATModel3Pooled(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=4, dropout=0.5, pooling_type='sag')
    model1.load_state_dict(torch.load('./experiments/GATModel3Pooled/experiment0SAG/models/model_10.pt'))
    model1.eval()

    model2 = GINEModel5Pooled(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5, pooling_type='topk')
    model2.load_state_dict(torch.load('./experiments/GINEModel5Pooled/experiment1Topk/models/model_3.pt'))
    model2.eval()

    model3 = NNModel2SAG(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5)
    model3.load_state_dict(torch.load('./experiments/NNModel2SAG/experiment0/models/model_0.pt'))
    model3.eval()

    save_dir = 'dataset/embeddings_fusion/AllFused1'
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='dataset/')
    fingerprints = torch.load('./dataset/fingerprints/fingerprints_list.pt')

    atom_encoder = AtomEncoder(emb_dim=64)
    bond_encoder = BondEncoder(emb_dim=32)

    dataset = list(DataLoader(dataset, batch_size=1))

    fused_embeddings = []
    labels = []

    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        fingerprint = fingerprints[idx]
        embedding1 = model1.calculate_embedding(atom_encoder(data.x), data.edge_index, bond_encoder(data.edge_attr), data.batch).squeeze()
        embedding2 = model2.calculate_embedding(atom_encoder(data.x), data.edge_index, bond_encoder(data.edge_attr), data.batch).squeeze()
        embedding3 = model3.calculate_embedding(atom_encoder(data.x), data.edge_index, bond_encoder(data.edge_attr), data.batch).squeeze()
        fused_embeddings.append(torch.cat((embedding1, embedding2, embedding3, fingerprint), dim=0).detach())
        labels.append(data.y)


    torch.save(fused_embeddings, f'{save_dir}/fused_embeddings.pt')
    torch.save(labels, f'{save_dir}/labels.pt')
    print('Embeddings saved.')

    print(f'Length of fused embedding: {len(fused_embeddings[0])}')


