from torch.utils.data import DataLoader
import torch

from datasets import FusionDataset
from models import FusionModel1
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from ogb.graphproppred import Evaluator

from tqdm import tqdm
import os


def evaluate_epoch(model, val_loader, loss_fn, device, threshold):
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
        for embeddings, labels in val_loader:
            # Forward pass
            embeddings, labels = embeddings.to(device), labels.to(device)
            out = model(embeddings)
            loss = loss_fn(out, labels.float().squeeze(1))

            # Updating Metrics
            pred = (out > threshold).float().squeeze()
            true = labels.squeeze()
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


def test_models(model, results_folder_path, embeddings_folder):
    # Loading dataset
    fused_embeddings_path = f'{embeddings_folder}/fused_embeddings.pt'
    labels_path = f'{embeddings_folder}/labels.pt'
    testing_dataset = FusionDataset(mode='test', embeddings_path=fused_embeddings_path, labels_path=labels_path)
    test_loader = DataLoader(testing_dataset, batch_size=256, shuffle=False)

    # Setting up device to be used
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    model.to(device)

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
            evaluate_epoch(model, test_loader, criterion, device, clf_threshold)
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
