import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score
from tqdm import tqdm
import os
from data_pt import *
from model import DcSE as Mymodel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, train_loader, optimizer, criterion, device, label_smoothing=0.0):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        if label_smoothing > 0:
            labels = labels.float() * (1 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def get_metrics(y_true, y_prob):
    y_pred = y_prob >= 0.5
    return [
        round(accuracy_score(y_true, y_pred), 4),
        round(roc_auc_score(y_true, y_prob), 4),
        round(average_precision_score(y_true, y_prob), 4),
        round(recall_score(y_true, y_pred, zero_division=0), 4),
        round(precision_score(y_true, y_pred, zero_division=0), 4),
    ]


def cross_validation(split_id, batch_size, lr, epoch, k_folds=5, species="human", label_smoothing=0.05):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, _ = load_datasets(f"./data/datasets/datasets_{species}_split_{split_id}.pt")
    labels = np.array([item[1] for item in dataset])

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    base_dir = f"./Demo_split_{split_id}"
    os.makedirs(f"{base_dir}/result", exist_ok=True)
    os.makedirs(f"{base_dir}/{species}_model", exist_ok=True)

    result_file = f"{base_dir}/result/{species}_results.txt"
    with open(result_file, "w") as f:
        f.write("Fold\tBestEpoch\tAcc\tAUC\tAUPR\tRecall\tPrecision\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n=== Split {split_id} | Fold {fold + 1}/{k_folds} ===")

        model = Mymodel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_metrics = {"epoch": 0, "acc": 0.0}
        fold_model_dir = f"{base_dir}/{species}_model/fold_{fold + 1}"
        os.makedirs(fold_model_dir, exist_ok=True)

        epoch_metrics_file = f"{fold_model_dir}/epoch_metrics.txt"
        with open(epoch_metrics_file, "w") as ef:
            ef.write("Epoch\tAcc\tAUC\tAUPR\tRecall\tPrecision\n")

        for e in range(epoch):
            train_loss = train(
                model,
                DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True),
                optimizer,
                criterion,
                device,
                label_smoothing=label_smoothing,
            )

            print(f"Epoch {e + 1}/{epoch}, Loss: {train_loss:.4f}")
            y_true, y_prob = evaluate(model, DataLoader(Subset(dataset, val_idx), batch_size=batch_size), device)
            acc, auc, aupr, recall, precision = get_metrics(y_true, y_prob)


            with open(epoch_metrics_file, "a") as ef:
                ef.write(f"{e + 1}\t{acc}\t{auc}\t{aupr}\t{recall}\t{precision}\n")


            if acc > best_metrics["acc"]:
                best_metrics.update(
                    {"epoch": e + 1, "acc": acc, "auc": auc, "aupr": aupr, "recall": recall, "precision": precision}
                )
                torch.save(model.state_dict(), f"{fold_model_dir}/best.pth")


        torch.save(model.state_dict(), f"{fold_model_dir}/final.pth")


        with open(result_file, "a") as f:
            f.write(
                f"{fold + 1}\t{best_metrics['epoch']}\t{best_metrics['acc']}\t"
                f"{best_metrics['auc']}\t{best_metrics['aupr']}\t"
                f"{best_metrics['recall']}\t{best_metrics['precision']}\n"
            )

    print(f"Split {split_id} 完成！")


if __name__ == "__main__":

    from concurrent.futures import ProcessPoolExecutor

    species_all = ["human", "mouse"]

    for species_id in species_all:

        def run_single_split(split_id):
            cross_validation(
                split_id=split_id, batch_size=64, lr=1e-4, epoch=30, k_folds=5, species=species_id
            )  

        for split_id in range(10):
            run_single_split(split_id)
