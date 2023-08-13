from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from models import GATModel
from tqdm import tqdm
import os
from training import evaluate_epoch


def test_models(model, results_folder_path):
    # Loading dataset
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='dataset/')
    print(f'Total {len(dataset)} graphs in dataset.')

    # Creating training and validation data loaders
    split_idx = dataset.get_idx_split()
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=64, shuffle=True)
    print(f'Testing dataset loaded. Size: {len(dataset[split_idx["test"]])} graphs.')

    # Setting up device to be used
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    model.to(device)

    # Initializing atom and bond encoders
    atom_encoder = AtomEncoder(emb_dim=64).to(device)
    bond_encoder = BondEncoder(emb_dim=32).to(device)

    # Creating files to write metrics
    if not os.path.exists(results_folder_path):
        print('Specified folder does not exist.')
        return
    testing_folder_path = os.path.join(results_folder_path, 'tests')
    os.mkdir(testing_folder_path)
    models_path = os.path.join(results_folder_path, 'models')
    all_models = os.listdir(models_path)

    # Initializing criterion and threshold
    criterion = torch.nn.BCELoss()
    clf_threshold = 0.5

    max_auc_roc = 0.0
    best_model = None

    # Models iteration
    for model_name in tqdm(all_models):

        model_path = os.path.join(models_path, model_name)
        results_file_path = os.path.join(testing_folder_path, model_name[:-2] + 'txt')
        open(results_file_path, 'w').close()
        model.load_state_dict(torch.load(model_path))

        # Evaluate
        testing_loss, testing_accuracy, testing_precision, testing_recall, testing_f1, testing_auc_roc = \
            evaluate_epoch(model, test_loader, criterion, device, atom_encoder, bond_encoder, clf_threshold)
        print(f'\nTesting model {model_name}: average loss = {testing_loss}, F1 = {testing_f1}, AUC ROC = {testing_auc_roc}')
        with open(results_file_path, 'a') as file:
            file.write(f'{model_name},{testing_loss},{testing_accuracy},{testing_precision},{testing_recall},{testing_f1},{testing_auc_roc}\n')
        if testing_auc_roc > max_auc_roc:
            best_model = model_name
            max_auc_roc = testing_auc_roc

    print(f'Best model: {best_model}: AUC ROC = {max_auc_roc}')


if __name__ == '__main__':
    model = GATModel(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=2, dropout=0.5)
    test_models(model, './experiment0')
