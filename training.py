from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from torch.optim import Adam
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from models import GATModel
from tqdm import tqdm
import os


def train_epoch(model, train_loader, optimizer, loss_fn, device, atom_encoder, bond_encoder, threshold):
    model.train()
    # Initializing Metrics
    losses = torch.tensor([], requires_grad=False, device=device)
    accuracy = BinaryAccuracy(threshold=threshold).to(device)
    precision = BinaryPrecision(threshold=threshold).to(device)
    recall = BinaryRecall(threshold=threshold).to(device)
    f1 = BinaryF1Score(threshold=threshold).to(device)
    auc_roc = BinaryAUROC().to(device)

    # Batch iteration
    for data in train_loader:
        if data.y.shape == torch.Size([]):
            print("Empty batch detected.")
        # Forward and backward pass
        data = data.to(device)
        optimizer.zero_grad()
        out = model(atom_encoder(data.x), data.edge_index, bond_encoder(data.edge_attr), data.batch)
        loss = loss_fn(out, data.y.float())
        loss.backward()
        optimizer.step()

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


def train(model, num_epochs, results_folder_path):
    # Loading dataset
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='dataset/')
    print(f'Total {len(dataset)} graphs in dataset.')

    # Creating training and validation data loaders
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=64, shuffle=True)
    print(f'Training dataset loaded. Size: {len(dataset[split_idx["train"]])} graphs.')
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=64, shuffle=True)
    print(f'Validation dataset loaded.Size: {len(dataset[split_idx["valid"]])} graphs')

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
        os.mkdir(results_folder_path)
    training_file_path = os.path.join(results_folder_path, 'training.csv')
    open(training_file_path, 'w').close()
    validation_file_path = os.path.join(results_folder_path, 'validation.csv')
    open(validation_file_path, 'w').close()
    os.mkdir(os.path.join(results_folder_path, 'models'))

    # Initializing optimizer and criterion
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss()
    clf_threshold = 0.5
    max_auc_roc_training, max_auc_roc_validation = 0, 0
    should_save_model = False

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        # Train
        training_loss, training_accuracy, training_precision, training_recall, training_f1, training_auc_roc = \
            train_epoch(model, train_loader, optimizer, criterion, device, atom_encoder, bond_encoder, clf_threshold)
        print(f'\nEpoch {epoch} Training: average loss = {training_loss}, accuracy = {training_accuracy}, F1 = {training_f1}, AUC ROC = {training_auc_roc}')
        with open(training_file_path, 'a') as file:
            file.write(f'{epoch},{training_loss},{training_accuracy},{training_precision},{training_recall},{training_f1},{training_auc_roc}\n')
        if training_auc_roc > max_auc_roc_training:
            should_save_model = True
            max_auc_roc_training = training_auc_roc

        # Validate
        validation_loss, validation_accuracy, validation_precision, validation_recall, validation_f1, validation_auc_roc = \
            evaluate_epoch(model, val_loader, criterion, device, atom_encoder, bond_encoder, clf_threshold)
        print(f'\nEpoch {epoch} Validation: average loss = {validation_loss}, accuracy = {validation_accuracy}, F1 = {validation_f1}, AUC ROC = {validation_auc_roc}')
        with open(validation_file_path, 'a') as file:
            file.write(f'{epoch},{validation_loss},{validation_accuracy},{validation_precision},{validation_recall},{validation_f1},{validation_auc_roc}\n')
        if validation_auc_roc > max_auc_roc_validation:
            should_save_model = True
            max_auc_roc_validation = validation_auc_roc

        # Save model
        if should_save_model or epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(results_folder_path, f'models/model_{epoch}.pt'))
            should_save_model = False

    # Save final model
    torch.save(model.state_dict(), os.path.join(results_folder_path, f'models/model_{num_epochs}.pt'))


if __name__ == '__main__':
    model = GATModel()
    train(model=model, num_epochs=1000, results_folder_path='experiment2')
