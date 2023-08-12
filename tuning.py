import optuna
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.nn import Linear
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from models import GATModelTune

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_fn(out, data.y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = (out > 0.5).float()
        correct += correct_predictions(pred, data.y).sum().item()
        total += data.num_graphs

    accuracy = correct / total
    average_loss = total_loss / len(train_loader)
    return average_loss, accuracy

def evaluate(model, loader, device):
    model.eval()
    # ... Your evaluation logic ...

# Define your objective function for Optuna
def objective(trial):
    input_dim = 24  # Your input dimension
    hidden_channels = trial.suggest_int('hidden_channels', 32, 128)
    edge_dim = 24  # Your edge dimension
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    model = GATModelTune(input_dim, hidden_channels, edge_dim, dropout)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your dataset using DataLoader
    train_loader = DataLoader(...)  # Define your DataLoader here
    val_loader = DataLoader(...)    # Define your DataLoader here

    for epoch in range(epochs):
        train_loss = train(model, optimizer, train_loader, device)
        val_loss = evaluate(model, val_loader, device)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

# Create an Optuna study and optimize the objective function
if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


