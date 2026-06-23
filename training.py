import torch
import torch.nn as nn
from numpy import interp
import numpy as np

from dataprocess import kfoldprepare, load_static_embeddings
from model_ablation import MIRAGE_Ablation
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train(model, device, train_loader, data_o, optimizer, epoch, gamma):
    model.train()
    bce_loss_fn = nn.BCELoss()

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        data_o = data_o.to(device)

        output, orth_loss, _, _ = model(data, data_o)

        labels = data.y.view(-1, 1).float().to(device)
        task_loss = bce_loss_fn(output, labels)
        total_loss = task_loss + gamma * orth_loss

        total_loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f} (Task: {:.4f}, Orth: {:.4f})'.format(
                epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss.item(), task_loss.item(), orth_loss.item()))


def predicting(model, device, loader, data_o):
    model.eval()
    total_probs = []
    total_preds = []
    total_labels = []

       if __name__ == '__main__':
    modeling = MIRAGE_Ablation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    LR = 0.0005
    LOG_INTERVAL = 40
    NUM_EPOCHS = 25
    NUM_RUNS = 5
    GAMMA = 0.01
    EMBEDDING_PATH = 'Predataprocess/miRNA_embeddings.txt'
    EMBEDDING_DIM = 512

    print(f"=====================================")
    print(f"🌟 当前开始实验组别：{MODEL_NAME}")
    print(f"=====================================")

    mirna_pretrained_feats = load_static_embeddings(EMBEDDING_PATH)
    train_data, test_data, data_o = kfoldprepare()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)

    accuracies, precisions, recalls, f1_scores, roc_aucs, pr_aucs = [], [], [], [], [], []

    epoch_history = []
    drug_focus_history = []
    mirna_focus_history = []

    for fold in range(NUM_RUNS):
        print(f"\nFold {fold + 1}/5")

        train_loader = DataLoader(train_data[fold], batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data[fold], batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

        model = modeling(
            pretrained_embeddings=mirna_pretrained_feats,
            embedding_dim=EMBEDDING_DIM,
            proj_dim=256,
            use_drug_seq=use_drug_seq,
            use_drug_graph=use_drug_graph,
            use_mirna_seq=use_mirna_seq,
            use_mirna_gip=use_mirna_gip,
            use_spd_attn=use_spd_attn,
            use_bipartite=use_bipartite
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, data_o, optimizer, epoch + 1, gamma=GAMMA)
            print(f"--- Epoch {epoch + 1} Evaluation ---")

             acc, pre, rec, f1, roc, pr, fpr, tpr, r_vals, p_vals, avg_d, avg_m = predicting(model, device, test_loader,
                                                                                            data_o)    
        print("\nMake prediction for {} samples...".format(len(test_loader.dataset)))
        accuracy, precision, recall, f1, roc_auc, pr_auc, fpr, tpr, recall_vals, precision_vals, _, _ = predicting(
            model, device, test_loader, data_o)

        sorted_indices = np.argsort(recall_vals)
        recall_vals_sorted = recall_vals[sorted_indices]
        precision_vals_sorted = precision_vals[sorted_indices]

        interpolated_precision = interp(mean_recall, recall_vals_sorted, precision_vals_sorted)
        mean_precision += interpolated_precision

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

        sorted_indices_roc = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices_roc]
        tpr_sorted = tpr[sorted_indices_roc]
        mean_tpr += interp(mean_fpr, fpr_sorted, tpr_sorted)

    mean_tpr[0] = 0.0

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_aucs)
    avg_pr_auc = np.mean(pr_aucs)

    std_accuracy = np.std(accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1_scores)
    std_roc_auc = np.std(roc_aucs)
    std_pr_auc = np.std(pr_aucs)

    mean_precision /= NUM_RUNS
    mean_tpr /= NUM_RUNS
    mean_tpr[-1] = 1.0

    mean_fpr_tpr = np.vstack((mean_fpr, mean_tpr)).T
    np.savetxt(f'{MODEL_NAME}_mean_fpr_tpr.csv', mean_fpr_tpr, delimiter=',', header='mean_fpr,mean_tpr', comments='')
    mean_recall_precision = np.vstack((mean_recall, mean_precision)).T
    np.savetxt(f'{MODEL_NAME}_mean_recall_precision.csv', mean_recall_precision, delimiter=',',
               header='mean_recall,mean_precision',
               comments='')

    print("\nAverage Metrics after 5-Fold Cross-Validation:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"ROC AUC: {avg_roc_auc:.4f}")
    print(f"PR AUC: {avg_pr_auc:.4f}")
    print(f"\n🎉 {MODEL_NAME} 实验完美收官！CSV数据和 .pth 权重文件均已就绪。")