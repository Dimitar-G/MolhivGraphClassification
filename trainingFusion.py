from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch
from torch.optim import Adam
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from models import FusionModel1, FusionModel2, FusionModel3
from tqdm import tqdm
import os
from datasets import FusionDataset
from testingFusion import test_models, evaluate_epoch
import numpy as np
from plots import generate_plots


def train_epoch(model, train_loader, optimizer, loss_fn, device, threshold):
    model.train()
    # Initializing Metrics
    losses = torch.tensor([], requires_grad=False, device=device)
    accuracy = BinaryAccuracy(threshold=threshold).to(device)
    precision = BinaryPrecision(threshold=threshold).to(device)
    recall = BinaryRecall(threshold=threshold).to(device)
    f1 = BinaryF1Score(threshold=threshold).to(device)
    auc_roc = BinaryAUROC().to(device)

    # Batch iteration
    for embeddings, labels in train_loader:
        # Forward and backward pass
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(embeddings)
        loss = loss_fn(out, labels.float().squeeze(1))
        loss.backward()
        optimizer.step()

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


def train(model, num_epochs, embeddings_folder, results_folder_path, learning_rate=0.0001, batch_size=64, weighted_sampling=False):
    # Loading dataset
    fused_embeddings_path = f'{embeddings_folder}/fused_embeddings.pt'
    labels_path = f'{embeddings_folder}/labels.pt'
    training_dataset = FusionDataset(mode='train', embeddings_path=fused_embeddings_path, labels_path=labels_path)
    validation_dataset = FusionDataset(mode='valid', embeddings_path=fused_embeddings_path, labels_path=labels_path)

    # Creating training and validation data loaders
    if weighted_sampling:
        labels = [int(item[1]) for item in training_dataset]
        labels_unique, counts = np.unique(labels, return_counts=True)
        class_weights = [1 / c for c in counts]
        label_weights = [class_weights[e] for e in labels]
        sampler = WeightedRandomSampler(label_weights, len(labels))
        train_loader = DataLoader(training_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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
        os.mkdir(results_folder_path)
    training_file_path = os.path.join(results_folder_path, 'training.csv')
    open(training_file_path, 'w').close()
    validation_file_path = os.path.join(results_folder_path, 'validation.csv')
    open(validation_file_path, 'w').close()
    info_file_path = os.path.join(results_folder_path, 'info.txt')
    open(info_file_path, 'w').close()
    os.mkdir(os.path.join(results_folder_path, 'models'))

    # Initializing optimizer and criterion
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    clf_threshold = 0.5
    max_auc_roc_training, max_auc_roc_validation = 0, 0
    should_save_model = False

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        # Train
        training_loss, training_accuracy, training_precision, training_recall, training_f1, training_auc_roc = \
            train_epoch(model, train_loader, optimizer, criterion, device, clf_threshold)
        print(f'\nEpoch {epoch} Training: average loss = {training_loss}, F1 = {training_f1}, AUC ROC = {training_auc_roc}')
        with open(training_file_path, 'a') as file:
            file.write(f'{epoch},{training_loss},{training_accuracy},{training_precision},{training_recall},{training_f1},{training_auc_roc}\n')
        if training_auc_roc > max_auc_roc_training:
            # should_save_model = True
            max_auc_roc_training = training_auc_roc

        # Validate
        validation_loss, validation_accuracy, validation_precision, validation_recall, validation_f1, validation_auc_roc = \
            evaluate_epoch(model, val_loader, criterion, device, clf_threshold)
        print(f'Epoch {epoch} Validation: average loss = {validation_loss}, F1 = {validation_f1}, AUC ROC = {validation_auc_roc}')
        with open(validation_file_path, 'a') as file:
            file.write(f'{epoch},{validation_loss},{validation_accuracy},{validation_precision},{validation_recall},{validation_f1},{validation_auc_roc}\n')
        if validation_auc_roc > max_auc_roc_validation:
            should_save_model = True
            max_auc_roc_validation = validation_auc_roc

        # Save model
        if should_save_model or epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(results_folder_path, f'models/model_{epoch}.pt'))
            should_save_model = False

    # Save final model
    torch.save(model.state_dict(), os.path.join(results_folder_path, f'models/model_{num_epochs}.pt'))
    print(f'TRAINING: Max AUC ROC = {max_auc_roc_training}')
    print(f'VALIDATION: Max AUC ROC = {max_auc_roc_validation}')
    with open(info_file_path, 'a') as file:
        file.write(f'\nTRAINING: Max AUC ROC = {max_auc_roc_training}')
        file.write(f'\nVALIDATION: Max AUC ROC = {max_auc_roc_validation}')
        file.flush()


if __name__ == '__main__':

    # print('Training FusionModel1:')
    # model = FusionModel1(input_dim=2471)
    # training_folder = './experiments_fusion/GINEModel5PooledTopK/experiment0'
    # embeddings_folder = './dataset/embeddings_fusion/GINEModel5TopK/experiment1TopK'
    # train(model=model, num_epochs=6, results_folder_path=training_folder, embeddings_folder=embeddings_folder, learning_rate=0.0001, batch_size=128, weighted_sampling=True)
    # generate_plots(training_folder)
    # test_models(model, training_folder, embeddings_folder=embeddings_folder)

    # print('Training FusionModel2:')
    # model = FusionModel2(input_dim=2471)
    # training_folder = './experiments_fusion/GINEModel5PooledTopK/experiment1'
    # embeddings_folder = './dataset/embeddings_fusion/GINEModel5TopK/experiment1TopK'
    # train(model=model, num_epochs=6, results_folder_path=training_folder, embeddings_folder=embeddings_folder, learning_rate=0.0001, batch_size=128, weighted_sampling=True)
    # generate_plots(training_folder)
    # test_models(model, training_folder, embeddings_folder=embeddings_folder)

    # print('Training FusionModel3:')
    # model = FusionModel3(input_dim=2471)
    # training_folder = './experiments_fusion/GINEModel5PooledTopK/experiment2'
    # embeddings_folder = './dataset/embeddings_fusion/GINEModel5TopK/experiment1TopK'
    # train(model=model, num_epochs=6, results_folder_path=training_folder, embeddings_folder=embeddings_folder, learning_rate=0.0001, batch_size=128, weighted_sampling=True)
    # generate_plots(training_folder)
    # test_models(model, training_folder, embeddings_folder=embeddings_folder)

    # print('Training FusionModel3:')
    # model = FusionModel3(input_dim=8181)
    # training_folder = './experiments_fusion/AllFused/experiment2_1'
    # embeddings_folder = './dataset/embeddings_fusion/AllFused'
    # train(model=model, num_epochs=6, results_folder_path=training_folder, embeddings_folder=embeddings_folder, learning_rate=0.00001, batch_size=128, weighted_sampling=True)
    # generate_plots(training_folder)
    # test_models(model, training_folder, embeddings_folder=embeddings_folder)

    # print('Training FusionModel:')
    # model = FusionModel1(input_dim=3239)
    # training_folder = './experiments_fusion/GATModel3PooledSAG/experiment0_1'
    # embeddings_folder = './dataset/embeddings_fusion/GATModel3Pooled/experiment0SAG'
    # train(model=model, num_epochs=10, results_folder_path=training_folder, embeddings_folder=embeddings_folder, learning_rate=0.00001, batch_size=128, weighted_sampling=True)
    # generate_plots(training_folder)
    # test_models(model, training_folder, embeddings_folder=embeddings_folder)

    # print('Training FusionModelAll:')
    # model = FusionModel3(input_dim=3751)
    # training_folder = './experiments_fusion/AllFused/experiment5'
    # embeddings_folder = './dataset/embeddings_fusion/AllFused1'
    # train(model=model, num_epochs=6, results_folder_path=training_folder, embeddings_folder=embeddings_folder, learning_rate=0.00001, batch_size=128, weighted_sampling=True)
    # generate_plots(training_folder)
    # test_models(model, training_folder, embeddings_folder=embeddings_folder)

    # print('Training FusionModelAll:')
    # model = FusionModel1(input_dim=3751)
    # training_folder = './experiments_fusion/AllFused/experiment6_1cont'
    # embeddings_folder = './dataset/embeddings_fusion/AllFused1'
    # train(model=model, num_epochs=15, results_folder_path=training_folder, embeddings_folder=embeddings_folder, learning_rate=0.00001, batch_size=128, weighted_sampling=False)
    # generate_plots(training_folder)
    # test_models(model, training_folder, embeddings_folder=embeddings_folder)

    print('Training FusionModel3:')
    model = FusionModel1(input_dim=3751)
    training_folder = './experiments_fusion/AllFused/experiment0_2cont'
    embeddings_folder = './dataset/embeddings_fusion/AllFused1'
    train(model=model, num_epochs=50, results_folder_path=training_folder, embeddings_folder=embeddings_folder, learning_rate=0.00001, batch_size=128, weighted_sampling=False)
    generate_plots(training_folder)
    test_models(model, training_folder, embeddings_folder=embeddings_folder)

