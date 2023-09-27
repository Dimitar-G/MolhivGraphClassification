from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from models import GATModel, GATModelExtended, NNModel2, GATModel3Pooled, GATModel5, GATModelPlus, GATModel5SAG, GATModelSAG, GINEModel3, GINEModel3Pooled, GINEModel5, GINEModel5Pooled, NNModel3, NNModel2SAG, NNModel3Pooled
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from ogb.graphproppred import Evaluator

from tqdm import tqdm
import os


def evaluate_epoch_ogb(model, val_loader, atom_encoder, bond_encoder, threshold):
    model.eval()
    # Initializing Metrics
    evaluator = Evaluator(name='ogbg-molhiv')
    true_all = torch.tensor([])
    pred_all = torch.tensor([])

    # Batch iteration
    with torch.no_grad():
        for data in val_loader:
            if data.y.shape == torch.Size([]):
                print("Empty batch detected.")
            # Forward pass
            out = model(atom_encoder(data.x), data.edge_index, bond_encoder(data.edge_attr), data.batch)

            # Updating Metrics
            pred = (out > threshold).float()
            true = data.y

            true_all = torch.cat((true_all, true), dim=0)
            pred_all = torch.cat((pred_all, out), dim=0)

    # Computing Metrics
    results_dict = evaluator.eval({'y_true': true_all, 'y_pred': pred_all})
    return results_dict['rocauc']


def test_models_ogb(model, results_folder_path):
    # Loading dataset
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='dataset/')
    split_idx = dataset.get_idx_split()
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=256, shuffle=False)

    # Initializing atom and bond encoders
    atom_encoder = AtomEncoder(emb_dim=64)
    bond_encoder = BondEncoder(emb_dim=32)

    # Creating files to write metrics
    if not os.path.exists(results_folder_path):
        print('Specified folder does not exist.')
        return
    models_path = os.path.join(results_folder_path, 'models')
    all_models = os.listdir(models_path)
    all_models = sorted(all_models, key=lambda s: int(s[6:-3]))
    # Initializing criterion and threshold
    clf_threshold = 0.5

    max_auc_roc = 0.0
    best_model = None

    results_file_path = os.path.join(results_folder_path, 'testingOGB.csv')
    open(results_file_path, 'w').close()

    # Models iteration
    for model_name in tqdm(all_models):

        model_path = os.path.join(models_path, model_name)
        model.load_state_dict(torch.load(model_path))

        # Evaluate
        testing_auc_roc = evaluate_epoch_ogb(model, test_loader, atom_encoder, bond_encoder, clf_threshold)
        print(f'\nTesting model {model_name}: AUC ROC = {testing_auc_roc}')
        with open(results_file_path, 'a') as file:
            file.write(f'{model_name},{testing_auc_roc}\n')
            file.flush()
        if testing_auc_roc > max_auc_roc:
            best_model = model_name
            max_auc_roc = testing_auc_roc

    print(f'TESTING: Best model = {best_model}, AUC ROC = {max_auc_roc}')
    info_file_path = os.path.join(results_folder_path, 'info.txt')
    if os.path.exists(info_file_path):
        with open(info_file_path, 'a') as file:
            file.write(f'\nTESTING: Best model = {best_model}, AUC ROC = {max_auc_roc}')
            file.flush()


if __name__ == '__main__':
    model = GATModel(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=4, dropout=0.5)
    test_models_ogb(model, './experiments/GATModel/experiment1_1')

    model = GATModel(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=8, dropout=0.5)
    test_models_ogb(model, './experiments/GATModel/experiment3')

    model = GATModel3Pooled(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=4, dropout=0.5, pooling_type='sag')
    test_models_ogb(model, './experiments/GATModel3Pooled/experiment0SAG')

    model = GATModel3Pooled(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=4, dropout=0.5)
    test_models_ogb(model, './experiments/GATModel3Pooled/experiment0Topk')

    model = GATModel3Pooled(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=4, dropout=0.5, pooling_type='sag')
    test_models_ogb(model, './experiments/GATModel3Pooled/experiment1SAG')

    model = GATModel3Pooled(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=4, dropout=0.5, pooling_type='topk')
    test_models_ogb(model, './experiments/GATModel3Pooled/experiment1Topk')

    model = GATModel5(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=6, dropout=0.5)
    test_models_ogb(model, './experiments/GATModel5/experiment0')

    model = GATModel5(node_embedding_size=64, edge_embedding_size=32, hidden_channels=512, num_heads=4, dropout=0.5)
    test_models_ogb(model, './experiments/GATModel5/experiment2')

    model = GATModelSAG(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=8, dropout=0.5)
    test_models_ogb(model, './experiments/GATModelSAG/experiment1')
