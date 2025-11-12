import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DcSE as Mymodel  
from data_pt import *
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    recall_score,
    precision_score,
    matthews_corrcoef,
    f1_score,
)
from sklearn.calibration import calibration_curve
import warnings
import pandas as pd

warnings.filterwarnings("ignore")  


def multi_source_ensemble_predict(model_paths, test_data_path, device="cuda"):

    _, test_dataset = load_datasets(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    y_true = np.array([item[1] for item in test_dataset])


    models = []
    for config in model_paths:
        model = config["model_class"]().to(device)
        model.load_state_dict(torch.load(config["path"]))
        model.eval()
        models.append(
            {
                "DcSEResult": model,
                "weight": config.get("weight", 1.0),  
                "calibration": config.get("calibration", None),  
            }
        )
        print(f"Loaded {config['model_class'].__name__} from {config['path']}")


    all_probs = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Multi-DcSEResult Ensemble"):
            inputs = inputs.to(device)
            batch_probs = []

            for model_info in models:
                model = model_info["DcSEResult"]
                weight = model_info["weight"]


                outputs = model(inputs).squeeze()


                if model_info["calibration"] is not None:
                    outputs = model_info["calibration"](outputs)


                probs = outputs.cpu().numpy()


                batch_probs.append(probs * weight)

            avg_probs = np.sum(batch_probs, axis=0) / sum(m["weight"] for m in models)
            all_probs.extend(avg_probs)

    return np.array(all_probs), y_true


def evaluate_ensemble(y_true, y_probs, threshold=0.5, save_dir="./results", split_id=None):

    os.makedirs(save_dir, exist_ok=True)

    # 基础指标计算
    y_pred = (y_probs >= threshold).astype(int)
    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probs),
        "AUPR": average_precision_score(y_true, y_probs),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    filename = f"{save_dir}/mouse_metrics.txt"
    if split_id is not None:
        filename = f"{save_dir}/mouse_metrics_split_{split_id}.txt"

    with open(filename, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    return metrics


def compute_aggregate_stats(all_metrics):

    metric_names = list(all_metrics[0].keys())
    metrics_array = {name: [] for name in metric_names}
    
    for metrics in all_metrics:
        for name in metric_names:
            metrics_array[name].append(metrics[name])
    
    mean_metrics = {f"mean_{name}": np.mean(metrics_array[name]) for name in metric_names}
    std_metrics = {f"std_{name}": np.std(metrics_array[name]) for name in metric_names}
    

    aggregate_stats = {**mean_metrics, **std_metrics}
    return aggregate_stats, metrics_array



if __name__ == "__main__":

    species = "human"
    num_splits = 10  
    all_metrics = []  
    
    for i in range(num_splits):

        model_configs = [
            {
                "path": f"./Ablation/model/DenseNet_SimpleCBAM_split_{i}/{species}_model/fold_1/last.pth",
                "model_class": Mymodel,
                "weight": 1.0,
            },
            {
                "path": f"./Ablation/model/DenseNet_SimpleCBAM_split_{i}/{species}_model/fold_2/last.pth",
                "model_class": Mymodel,
                "weight": 1.0,
            },
            {
                "path": f"./Ablation/model/DenseNet_SimpleCBAM_split_{i}/{species}_model/fold_3/last.pth",
                "model_class": Mymodel,
                "weight": 1.0,
            },
            {
                "path": f"./Ablation/model/DenseNet_SimpleCBAM_split_{i}/{species}_model/fold_4/last.pth",
                "model_class": Mymodel,
                "weight": 1.0,
            },
            {
                "path": f"./Ablation/model/DenseNet_SimpleCBAM_split_{i}/{species}_model/fold_5/last.pth",
                "model_class": Mymodel,
                "weight": 1.0,
            }
        ]


        ensemble_probs, y_true = multi_source_ensemble_predict(
            model_paths=model_configs,
            test_data_path=f"./data/datasets/datasets_{species}_split_{i}.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


        result_dir = f"Ablation/model/DenseNet_SimpleCBAM_split_{i}/ensemble_results_new"
        os.makedirs(result_dir, exist_ok=True)
        
        metrics = evaluate_ensemble(y_true, ensemble_probs, save_dir=result_dir, split_id=i)
        all_metrics.append(metrics)  

        print(f"\n=== Split {i} Ensemble Model Performance ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    

    aggregate_stats, metrics_data = compute_aggregate_stats(all_metrics)
    

    summary_dir = f"Ablation/model/ensemble_summary_{species}"
    os.makedirs(summary_dir, exist_ok=True)


    with open(f"{summary_dir}/aggregate_metrics.txt", "w") as f:
        f.write(f"=== Average Metrics Across {num_splits} Splits ===\n")
        f.write("Mean Values:\n")
        for k, v in aggregate_stats.items():
            if k.startswith("mean_"):
                f.write(f"{k[5:]}: {v:.4f}\n")
        
        f.write("\nStandard Deviations:\n")
        for k, v in aggregate_stats.items():
            if k.startswith("std_"):
                f.write(f"{k[4:]}: {v:.4f}\n")
    

    data = []
    metric_names = list(all_metrics[0].keys())
    for i, metrics in enumerate(all_metrics):
        row = {"split": i}
        row.update({k: metrics[k] for k in metric_names})
        data.append(row)
    

    data_df = pd.DataFrame(data)
    data_df.to_csv(f"{summary_dir}/all_splits_results.csv", index=False)
    

    print(f"\n=== Average Metrics Across {num_splits} Splits ===")
    print("Mean Values:")
    for k, v in aggregate_stats.items():
        if k.startswith("mean_"):
            print(f"{k[5:]}: {v:.4f}")
    
    print("\nStandard Deviations:")
    for k, v in aggregate_stats.items():
        if k.startswith("std_"):
            print(f"{k[4:]}: {v:.4f}")
