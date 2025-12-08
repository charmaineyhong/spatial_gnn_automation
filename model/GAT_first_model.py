#!/usr/bin/env python3
"""
Binary GAT training script: DIMENSION vs No Annotation

For each element (node), predict whether it should get a DIMENSION
annotation or not.

Labels:
- 0: No dimension annotation
- 1: Has at least one DIMENSION annotation

Uses:
- GraphML (geometry + features)
- annotation.csv (existing annotations) → builds binary targets

Outputs:
- 5.OutputML_GAT_DIM/trained_model.pth
- 5.OutputML_GAT_DIM/trained_model.onnx
- 5.OutputML_GAT_DIM/per_class_metrics.csv
- Confusion matrices + plots (if visualize_graph.py is available)
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Optional plotting helpers (if available)
try:
    from visualize_graph import (
        plot_training_validation, plot_test_accuracy,
        plot_confusion_matrix as vg_plot_confusion_matrix,
        plot_prediction_distribution, plot_validation_accuracy,
        plot_lr_schedule, plot_class_accuracies
    )
except ImportError:
    vg_plot_confusion_matrix = None
    plot_training_validation = plot_test_accuracy = plot_prediction_distribution = None
    plot_validation_accuracy = plot_lr_schedule = plot_class_accuracies = None

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 100
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------------------
# Focal Loss (binary / multi-class)
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        inputs: logits (N, C)
        targets: long labels (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


# ----------------------------
# Light data augmentation
# ----------------------------
def data_augmentation(x, edge_index, training=True):
    if not training:
        return x, edge_index
    # Small Gaussian noise on node features
    if torch.rand(1).item() < 0.1:
        x = x + torch.randn_like(x) * 0.05
    # Randomly drop some edges
    if torch.rand(1).item() < 0.05 and edge_index.shape[1] > 0:
        keep_mask = torch.rand(edge_index.shape[1]) > 0.1
        if keep_mask.sum() > 0:
            edge_index = edge_index[:, keep_mask]
    return x, edge_index


# ----------------------------
# Helper: identify DIMENSION rows
# ----------------------------
def is_dimension_row(category, annotation_type):
    cat_u = str(category).upper()
    atype_u = str(annotation_type).upper()
    if "DIMENSION" in cat_u or "DIMENSIONS" in cat_u:
        return True
    if "DIM" in atype_u:
        return True
    return False


# ----------------------------
# Graph parsing: binary DIMENSION labels
# ----------------------------
def parse_graphml_to_pyg(graphml_path, extracted_base="../Extracted Data"):
    G = nx.read_graphml(graphml_path)

    base_name = os.path.basename(graphml_path).replace(".graphml", "").replace("PPVC_", "PPVC ")
    annotation_paths = [
        f"{extracted_base}/{base_name}, 20_Typ/annotation.csv",      # PPVC 01
        f"{extracted_base}/{base_name}_Typ 1/annotation.csv",        # PPVC 02, 04
        f"{extracted_base}/{base_name}, 16_Typ/annotation.csv",      # PPVC 05
        f"{extracted_base}/{base_name}_Typ/annotation.csv",          # PPVC 06, 22
        f"{extracted_base}/{base_name},31_Typ/annotation.csv",       # PPVC 21
        f"{extracted_base}/{base_name}_Typ1_testing/annotation.csv", # PPVC 03 (if exists)
    ]
    annotation_path = next((p for p in annotation_paths if os.path.exists(p)), None)

    dim_elements = set()

    if annotation_path:
        try:
            annotations_df = pd.read_csv(annotation_path)
            for _, row in annotations_df.iterrows():
                cat = row.get("category", "")
                atype = row.get("annotation_type", "")
                if not is_dimension_row(cat, atype):
                    continue
                target_ids_raw = str(row.get("target_element_ids", "")).split("|")
                for eid_str in target_ids_raw:
                    eid_str = eid_str.strip()
                    if not eid_str or eid_str == "nan":
                        continue
                    try:
                        eid = int(eid_str)
                    except ValueError:
                        continue
                    dim_elements.add(eid)
        except Exception as e:
            print(f"Warning: could not load annotations from {annotation_path}: {e}")
    else:
        print(f"Warning: No annotation file found for {graphml_path}")

    node_features = []
    node_labels = []
    node_mapping = {}

    for i, (node_id, data) in enumerate(G.nodes(data=True)):
        node_mapping[node_id] = i
        features = []

        # Geometric basics
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
        z = float(data.get("z", 0.0))
        w = float(data.get("width", 1.0))
        h = float(data.get("height", 1.0))
        d = float(data.get("depth", 1.0))
        features.extend([x, y, z, w, h, d])

        # Volume + surface area
        volume = w * h * d
        surface_area = 2 * (w * h + h * d + w * d)
        features.extend([volume, surface_area])

        # Ratios
        features.extend([
            w / h if h else 0.0,
            h / d if d else 0.0,
            w / d if d else 0.0,
        ])

        # Encoded material + type (from graphml attributes)
        material = data.get("material", "unknown")
        element_type = data.get("type", "unknown")
        material_encoding = {"concrete": 1, "steel": 2, "wood": 3, "glass": 4, "unknown": 0}
        type_encoding = {"wall": 1, "column": 2, "beam": 3, "slab": 4, "unknown": 0}
        features.extend([
            material_encoding.get(str(material).lower(), 0),
            type_encoding.get(str(element_type).lower(), 0),
        ])

        # Position again
        features.extend([x, y, z])
        node_features.append(features)

        # Label: Use label_type from GraphML if available, otherwise try to load from annotation file
        label = 0
        if "label_type" in data:
            # Use the label_type directly from GraphML
            try:
                label_type_val = int(data["label_type"])
                # label_type: 0=No Label, 1=Has Dimension
                label = 1 if label_type_val == 1 else 0
            except (ValueError, TypeError):
                label = 0
        else:
            # Fallback: check if element has a dimension annotation from external file
            try:
                node_int_id = int(data.get("node_id", node_id))
            except Exception:
                node_int_id = None

            if node_int_id is not None and node_int_id in dim_elements:
                label = 1
        
        node_labels.append(label)

    # Edges (undirected)
    edge_list = []
    for src, dst in G.edges():
        if src in node_mapping and dst in node_mapping:
            edge_list.append([node_mapping[src], node_mapping[dst]])
            edge_list.append([node_mapping[dst], node_mapping[src]])

    if not edge_list:
        print(f"Warning: No edges in {graphml_path}")
        return None

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x_tensor = torch.tensor(node_features, dtype=torch.float)
    y_tensor = torch.tensor(node_labels, dtype=torch.long)

    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor)


# ----------------------------
# GAT Model
# ----------------------------
class ImprovedGATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_classes=2, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim // num_heads, heads=num_heads,
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                            dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                            dropout=dropout, concat=True)
        self.gat4 = GATConv(hidden_dim, hidden_dim // 2, heads=num_heads // 2,
                            dropout=dropout, concat=True)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm((hidden_dim // 2) * (num_heads // 2))

        final_dim = (hidden_dim // 2) * (num_heads // 2)
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        x = self.dropout(self.ln1(F.elu(self.gat1(x, edge_index))))
        x = self.dropout(self.ln2(F.elu(self.gat2(x, edge_index))))
        x = self.dropout(self.ln3(F.elu(self.gat3(x, edge_index))))
        x = self.ln4(F.elu(self.gat4(x, edge_index)))
        return self.classifier(x)


# ----------------------------
# Train / Eval
# ----------------------------
def train_epoch(model, optimizer, criterion, loader, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_aug, edge_aug = data_augmentation(data.x, data.edge_index, training=True)
        out = model(x_aug, edge_aug)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        total_correct += (pred == data.y).sum().item()
        total_samples += data.y.size(0)

    return total_loss / len(loader), total_correct / total_samples


def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == data.y).sum().item()
            total_samples += data.y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    acc = total_correct / total_samples if total_samples else 0.0
    return acc, total_loss / len(loader), all_preds, all_labels


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    input_dir = "./3.InputML"
    output_dir = "./5.OutputML_GAT_DIM"
    os.makedirs(output_dir, exist_ok=True)

    graphml_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".graphml")]
    )
    if len(graphml_files) < 2:
        raise ValueError("Need at least 2 graphml files for cross validation.")

    data_list = []
    for fp in graphml_files:
        d = parse_graphml_to_pyg(fp)
        if d is not None:
            data_list.append(d)
    if len(data_list) < 2:
        raise ValueError("Parsed fewer than 2 valid graphs.")

    total_nodes = sum(len(d.y) for d in data_list)
    total_dim = sum((d.y == 1).sum().item() for d in data_list)
    print(f"Dataset: nodes={total_nodes}, dim_nodes={total_dim}, ratio={total_dim/total_nodes:.3f}")

    input_dim = data_list[0].x.shape[1]
    output_dim = 2
    label_names = ["No Annotation", "DIMENSION"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED) if len(data_list) >= 5 else LeaveOneOut()
    cv_name = "5-fold" if isinstance(splitter, KFold) else "leave-one-out"
    print(f"Cross-validation: {cv_name}")

    fold_accuracies, fold_losses = [], []
    all_y_true, all_y_pred = [], []
    best_fold_acc, best_state = -1, None

    for fold, (train_idx, test_idx) in enumerate(splitter.split(data_list), start=1):
        print(f"\n==== Fold {fold} ====")
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        model = ImprovedGATModel(
            input_dim=input_dim, hidden_dim=128, num_heads=8,
            num_classes=output_dim, dropout=0.3
        ).to(device)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=40)

        best_val_acc, patience_ctr, patience = 0.0, 0, 120
        epochs = 400
        train_losses, test_losses, test_accuracies, lrs = [], [], [], []

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_epoch(model, optimizer, criterion, train_loader, device)
            te_acc, te_loss, te_preds, te_labels = evaluate(model, criterion, test_loader, device)

            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            test_accuracies.append(te_acc)
            lrs.append(optimizer.param_groups[0]["lr"])

            scheduler.step(te_loss)

            if epoch % 10 == 0 or epoch <= 10:
                if len(set(te_labels)) > 1:
                    prec = precision_score(te_labels, te_preds, zero_division=0)
                    rec = recall_score(te_labels, te_preds, zero_division=0)
                    f1 = f1_score(te_labels, te_preds, zero_division=0)
                else:
                    prec = rec = f1 = 0.0
                print(
                    f"Fold {fold} | Epoch {epoch:03d} | "
                    f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.4f} | "
                    f"TestLoss={te_loss:.4f} TestAcc={te_acc:.4f} | "
                    f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} | LR={lrs[-1]:.6f}"
                )

            if te_acc > best_val_acc:
                best_val_acc, patience_ctr = te_acc, 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if plot_training_validation and epoch % 25 == 0:
                plot_training_validation(
                    train_losses, test_losses,
                    os.path.join(fold_dir, "training_validation.png")
                )
                plot_validation_accuracy(
                    test_accuracies,
                    os.path.join(fold_dir, "validation_accuracy.png")
                )
                plot_lr_schedule(lrs, os.path.join(fold_dir, "lr_schedule.png"))

        te_acc_final, te_loss_final, y_pred_fold, y_true_fold = evaluate(
            model, criterion, test_loader, device
        )
        fold_accuracies.append(te_acc_final)
        fold_losses.append(te_loss_final)
        all_y_true.extend(y_true_fold)
        all_y_pred.extend(y_pred_fold)

        # Confusion matrix per fold
        if vg_plot_confusion_matrix:
            vg_plot_confusion_matrix(
                np.array(y_true_fold), np.array(y_pred_fold),
                label_names, os.path.join(fold_dir, "confusion_matrix.png")
            )
        else:
            cm = confusion_matrix(y_true_fold, y_pred_fold, labels=range(output_dim))
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names
            )
            plt.title(f"Fold {fold} Confusion Matrix")
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"))
            plt.close()

        if plot_class_accuracies:
            plot_class_accuracies(
                np.array(y_true_fold), np.array(y_pred_fold),
                label_names, os.path.join(fold_dir, "class_accuracies_histogram.png")
            )

        if te_acc_final > best_fold_acc:
            best_fold_acc = te_acc_final
            best_state = model.state_dict()

        print(f"Fold {fold} Final Test Accuracy: {te_acc_final:.4f}")

    # ---- CV summary ----
    fold_accuracies = np.array(fold_accuracies)
    fold_losses = np.array(fold_losses)
    print("\n==== CV SUMMARY ====")
    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {i}: Acc={acc:.4f}")
    print(f"Mean Acc: {fold_accuracies.mean():.4f} | Std: {fold_accuracies.std():.4f}")

    # ---- Save best model ----
    best_model = ImprovedGATModel(
        input_dim=input_dim, hidden_dim=128, num_heads=8,
        num_classes=output_dim, dropout=0.3
    ).to(device)
    best_model.load_state_dict(best_state)
    pth_path = os.path.join(output_dir, "trained_model.pth")
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": 128,
            "heads": 8,
            "dropout": 0.3,
            "best_fold_acc": best_fold_acc,
            "cv_mean_acc": fold_accuracies.mean(),
            "cv_std_acc": fold_accuracies.std(),
            "seed": SEED,
            "cv_mode": cv_name,
            "label_names": label_names,
            "task_type": "binary_dimension_prediction",
        },
        pth_path,
    )
    print("Saved best PyTorch weights to:", pth_path)

    # ---- ONNX export (optional) ----
    try:
        onnx_path = os.path.join(output_dir, "trained_model.onnx")
        dummy_x = torch.randn(4, input_dim).to(device)
        dummy_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device)
        torch.onnx.export(
            best_model,
            (dummy_x, dummy_edge),
            onnx_path,
            input_names=["x", "edge_index"],
            output_names=["logits"],
            opset_version=16,
            dynamic_axes={
                "x": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "logits": {0: "num_nodes"},
            },
        )
        print("Saved ONNX model to:", onnx_path)
    except Exception as e:
        print(f"Warning: ONNX export failed (skipping): {e}")

    # ---- Overall metrics ----
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=range(output_dim), zero_division=0
    )
    metrics_df = pd.DataFrame(
        {"Class": label_names, "Precision": precision, "Recall": recall, "F1-Score": f1}
    )
    metrics_csv = os.path.join(output_dir, "per_class_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print("Per-class metrics saved to:", metrics_csv)
    print(metrics_df)

    if vg_plot_confusion_matrix:
        vg_plot_confusion_matrix(
            all_y_true, all_y_pred, label_names,
            os.path.join(output_dir, "confusion_matrix_overall.png")
        )
    else:
        cm = confusion_matrix(all_y_true, all_y_pred, labels=range(output_dim))
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names
        )
        plt.title("Overall Confusion Matrix")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_overall.png"))
        plt.close()

    if plot_class_accuracies:
        plot_class_accuracies(
            all_y_true, all_y_pred, label_names,
            os.path.join(output_dir, "class_accuracies_histogram_overall.png")
        )
