import os.path

import pandas as pd
import matplotlib.pyplot as plt


def generate_plots(metrics_folder_path):
    training = pd.read_csv(os.path.join(metrics_folder_path, 'training.csv'))
    validation = pd.read_csv(os.path.join(metrics_folder_path, 'validation.csv'))

    plots_folder = os.path.join(metrics_folder_path, 'plots')
    if not os.path.exists(plots_folder):
        os.mkdir(plots_folder)

    # Training loss
    plt.plot(training.iloc[:, 0], training.iloc[:, 1])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(plots_folder, 'training_loss.png'))
    plt.clf()

    # Validation loss
    plt.plot(validation.iloc[:, 0], validation.iloc[:, 1])
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(plots_folder, 'validation_loss.png'))
    plt.clf()

    # Joint loss
    plt.plot(training.iloc[:, 0], training.iloc[:, 1], label='Training Loss')
    plt.plot(validation.iloc[:, 0], validation.iloc[:, 1], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(plots_folder, 'joint_loss.png'))
    plt.clf()

    # Training F1 score
    plt.plot(training.iloc[:, 0], training.iloc[:, 5])
    plt.title('Training F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.savefig(os.path.join(plots_folder, 'training_f1.png'))
    plt.clf()

    # Validation F1 score
    plt.plot(validation.iloc[:, 0], validation.iloc[:, 5])
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.savefig(os.path.join(plots_folder, 'validation_f1.png'))
    plt.clf()

    # Joint F1 score
    plt.plot(training.iloc[:, 0], training.iloc[:, 5], label='Training F1-score')
    plt.plot(validation.iloc[:, 0], validation.iloc[:, 5], label='Validation F1-score')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.savefig(os.path.join(plots_folder, 'joint_f1.png'))
    plt.clf()

    # Training AUC ROC
    plt.plot(training.iloc[:, 0], training.iloc[:, 6])
    plt.title('Training AUC ROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC ROC')
    plt.savefig(os.path.join(plots_folder, 'training_aucroc.png'))
    plt.clf()

    # Validation AUC ROC
    plt.plot(validation.iloc[:, 0], validation.iloc[:, 6])
    plt.title('Validation AUC ROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC ROC')
    plt.savefig(os.path.join(plots_folder, 'validation_aucroc.png'))
    plt.clf()

    # Joint AUC ROC
    plt.plot(training.iloc[:, 0], training.iloc[:, 6], label='Training AUC ROC')
    plt.plot(validation.iloc[:, 0], validation.iloc[:, 6], label='Validation AUC ROC')
    plt.title('AUC ROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC ROC')
    plt.savefig(os.path.join(plots_folder, 'joint_aucroc.png'))
    plt.clf()


if __name__ == '__main__':
    generate_plots('./experiments/GATModel5/experiment0')
