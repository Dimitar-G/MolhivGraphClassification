from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from models import GATModel, GATModelExtended, NNModel2
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from ogb.graphproppred import Evaluator

from tqdm import tqdm
import os


def evaluate_epoch(model, val_loader, loss_fn, device, atom_encoder, bond_encoder, threshold):
    model.eval()
    # Initializing Metrics
    losses = torch.tensor([], requires_grad=False, device=device)
    accuracy = BinaryAccuracy(threshold=threshold).to(device)
    precision = BinaryPrecision(threshold=threshold).to(device)
    recall = BinaryRecall(threshold=threshold).to(device)
    f1 = BinaryF1Score(threshold=threshold).to(device)
    auc_roc = BinaryAUROC().to(device)

    # Batch iteration
    with torch.no_grad():
        for data in val_loader:
            if data.y.shape == torch.Size([]):
                print("Empty batch detected.")
            # Forward pass
            data = data.to(device)
            out = model(atom_encoder(data.x), data.edge_index, bond_encoder(data.edge_attr), data.batch)
            loss = loss_fn(out, data.y.float())

            # Updating Metrics
            pred = (out > threshold).float().squeeze()
            true = data.y.squeeze()
            losses = torch.cat((losses, loss.reshape(1)), dim=0)
            accuracy.update(input=pred, target=true)
            precision.update(input=pred, target=true)
            recall.update(input=pred, target=true)
            f1.update(input=pred, target=true)
            auc_roc.update(input=out.squeeze(), target=true)

    # Computing Metrics
    losses = torch.mean(losses, dim=0).detach().cpu()
    accuracy = accuracy.compute().detach().cpu()
    precision = precision.compute().detach().cpu()
    recall = recall.compute().detach().cpu()
    f1 = f1.compute().detach().cpu()
    auc_roc = auc_roc.compute().detach().cpu()
    return losses, accuracy, precision, recall, f1, auc_roc


def test_models(model, results_folder_path):
    # Loading dataset
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='dataset/')
    split_idx = dataset.get_idx_split()
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=256, shuffle=False)

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
    models_path = os.path.join(results_folder_path, 'models')
    all_models = os.listdir(models_path)
    all_models = sorted(all_models, key=lambda s: int(s[6:-3]))
    # Initializing criterion and threshold
    criterion = torch.nn.BCELoss()
    clf_threshold = 0.5

    max_auc_roc = 0.0
    best_model = None

    results_file_path = os.path.join(results_folder_path, 'testing.csv')
    open(results_file_path, 'w').close()

    # Models iteration
    for model_name in tqdm(all_models):

        model_path = os.path.join(models_path, model_name)
        model.load_state_dict(torch.load(model_path))

        # Evaluate
        testing_loss, testing_accuracy, testing_precision, testing_recall, testing_f1, testing_auc_roc = \
            evaluate_epoch(model, test_loader, criterion, device, atom_encoder, bond_encoder, clf_threshold)
        print(f'\nTesting model {model_name}: average loss = {testing_loss}, F1 = {testing_f1}, AUC ROC = {testing_auc_roc}')
        with open(results_file_path, 'a') as file:
            file.write(f'{model_name},{testing_loss},{testing_accuracy},{testing_precision},{testing_recall},{testing_f1},{testing_auc_roc}\n')
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
    model = NNModel2(node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5)
    test_models(model, './experiments/NNModel2/test_dir')
