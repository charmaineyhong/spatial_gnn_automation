#!/usr/bin/env python3
"""
GAT text detector (heuristic mapping)

Task:
  For each element/node, predict:
    0 = No Text
    1 = Has Text (any of the TextNotes we care about)

Key idea:
  - We treat annotation rows with category == "TextNotes" and non-empty value
    as "text annotations" (room labels, UNIT FFL, slab thickness, etc.).
  - For each such text row, we look at the (x,y) of the text and
    label ALL graph nodes within a radius R as Has Text.
  - If nothing falls within R, we still label the single nearest node,
    so every text row contributes at least one positive node.

Outputs (within ./5.OutputML_GAT_TEXT_HAS_HEURISTIC):
  - trained_model_text_has_heuristic.pth
  - trained_model_text_has_heuristic.onnx
  - per_class_metrics_text_has_heuristic.csv
  - confusion_matrix_overall_text_has_heuristic.png
  - Fold-wise plots if visualize_graph.py is available
"""

import os
import random
import math
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
        plot_training_validation,
        plot_test_accuracy,
        plot_confusion_matrix as vg_plot_confusion_matrix,
        plot_prediction_distribution,
        plot_validation_accuracy,
        plot_lr_schedule,
        plot_class_accuracies,
    )
except ImportError:
    vg_plot_confusion_matrix = None
    plot_training_validation = plot_test_accuracy = plot_prediction_distribution = \
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
# Labels
# ----------------------------
LABEL_NAMES = ["No Text", "Has Text"]
NUM_CLASSES = 2  # 0,1


# ----------------------------
# Focal loss with class weights
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        # inputs: (N, C) logits
        # targets: (N,) long
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# ----------------------------
# Light data augmentation
# ----------------------------
def data_augmentation(x, edge_index, training=True):
    if not training:
        return x, edge_index
    if torch.rand(1).item() < 0.1:
        x = x + torch.randn_like(x) * 0.05
    if torch.rand(1).item() < 0.05 and edge_index.shape[1] > 0:
        keep_mask = torch.rand(edge_index.shape[1]) > 0.1
        if keep_mask.sum() > 0:
            edge_index = edge_index[:, keep_mask]
    return x, edge_index


# ----------------------------
# Helper: which annotation rows count as "Text"
# ----------------------------
def is_text_umbrella(row):
    """
    Define the umbrella of text annotations we care about.

    For now: any TextNotes row with a non-empty value.
    (This includes room labels, UNIT FFL, slab thickness, etc.)
    We intentionally ignore:
      - Dimension rows (they're in another category)
      - Tags (Structural Framing Tags, Revision Cloud Tags, etc.)
      - Detail Items / Filled Regions / Openings
    """
    category = str(row.get("category", "")).strip().upper()
    value = str(row.get("value", "")).strip()
    if category != "TEXTNOTES":
        return False
    if value == "":
        return False
    # If later you want to exclude some weird text, you can add checks here.
    return True


# ----------------------------
# Graph parsing with heuristic text mapping
# ----------------------------
def parse_graphml_to_pyg(graphml_path, extracted_base="../Extracted Data",
                         text_radius=150.0):
    """
    Build a PyG Data object with:
      x: node features
      y: 0 (No Text) or 1 (Has Text), using heuristic XY mapping.

    text_radius: radius in the same units as node.x/node.y to label neighbours.
    """
    G = nx.read_graphml(graphml_path)

    # Find the matching annotation_with_targets.csv (augmented with mapped element IDs)
    base_name = os.path.basename(graphml_path).replace('.graphml', '').replace('PPVC_', 'PPVC ')
    annotation_paths = [
        f"{extracted_base}/{base_name}, 20_Typ/annotation_with_targets.csv",     # PPVC 01
        f"{extracted_base}/{base_name}_Typ 1/annotation_with_targets.csv",       # PPVC 02, 04
        f"{extracted_base}/{base_name}, 16_Typ/annotation_with_targets.csv",     # PPVC 05
        f"{extracted_base}/{base_name}_Typ/annotation_with_targets.csv",         # PPVC 06, 22
        f"{extracted_base}/{base_name},31_Typ/annotation_with_targets.csv",      # PPVC 21
        f"{extracted_base}/{base_name}_Typ1_testing/annotation_with_targets.csv" # PPVC 03
    ]
    annotation_path = next((p for p in annotation_paths if os.path.exists(p)), None)

    # --- Prepare nodes ---
    node_features = []
    node_xy = []    # (x,y) for each node, used for distance calc
    node_mapping = {}
    node_labels = []  # final labels: 0 or 1

    for i, (node_id, data) in enumerate(G.nodes(data=True)):
        node_mapping[node_id] = i
        feats = []

        # Basic geometry from GraphML (use correct attribute names)
        x = float(data.get("pos_x", 0.0))
        y = float(data.get("pos_y", 0.0))
        z = float(data.get("bb_zmin", 0.0))  # Use bounding box z min
        w = float(data.get("width", 1.0))
        h = float(data.get("height", 1.0))
        d = float(data.get("depth", 1.0))
        feats.extend([x, y, z, w, h, d])

        # Volume & surface area
        volume = w * h * d
        surface_area = 2 * (w * h + h * d + w * d)
        feats.extend([volume, surface_area])

        # Ratios
        feats.extend([
            w / h if h else 0.0,
            h / d if d else 0.0,
            w / d if d else 0.0
        ])

        # Encoded material & type
        material = data.get("material", "unknown")
        element_type = data.get("type", "unknown")
        material_encoding = {"concrete": 1, "steel": 2, "wood": 3, "glass": 4, "unknown": 0}
        type_encoding = {"wall": 1, "column": 2, "beam": 3, "slab": 4, "unknown": 0}
        feats.extend([
            material_encoding.get(str(material).lower(), 0),
            type_encoding.get(str(element_type).lower(), 0),
        ])

        # Position again (like your original script)
        feats.extend([x, y, z])

        node_features.append(feats)
        node_xy.append((x, y))
        node_labels.append(0)  # default: No Text

    num_nodes = len(node_features)
    if num_nodes == 0:
        print(f"Warning: graph {graphml_path} has 0 nodes.")
        return None

    # --- Use pre-computed target_element_ids from annotation_with_targets.csv ---
    if annotation_path:
        try:
            df = pd.read_csv(annotation_path)

            # Use the target_element_ids that were pre-computed by map_text_to_elements.py
            text_rows = df[df.apply(is_text_umbrella, axis=1)]
            print(f"[{os.path.basename(graphml_path)}] Text rows found: {len(text_rows)}")
            
            elements_with_text = set()
            for _, row in text_rows.iterrows():
                target_ids_str = str(row.get("target_element_ids", "")).strip()
                if target_ids_str and target_ids_str != "nan" and target_ids_str != "":
                    # Split by pipe to get multiple element IDs
                    for eid_str in target_ids_str.split("|"):
                        eid_str = eid_str.strip()
                        if eid_str and eid_str != "nan":
                            elements_with_text.add(eid_str)
            
            print(f"[{os.path.basename(graphml_path)}] Unique elements with text: {len(elements_with_text)}")
            
            # Now label nodes based on their node_id
            labeled_count = 0
            for i, (node_id, data) in enumerate(G.nodes(data=True)):
                if str(node_id) in elements_with_text:
                    node_labels[i] = 1
                    labeled_count += 1
            
            print(f"[{os.path.basename(graphml_path)}] Actually labeled {labeled_count} nodes as 'Has Text'")

        except Exception as e:
            print(f"Warning: Could not load annotations from {annotation_path}: {e}")
    else:
        print(f"Warning: No annotation file found for {graphml_path}")

    # --- Edges (undirected) ---
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

    # Quick debug stats for this graph
    num_text_nodes = int((y_tensor == 1).sum().item())
    print(f"[{os.path.basename(graphml_path)}] nodes={num_nodes}, HasText nodes={num_text_nodes}")

    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor)


# ----------------------------
# GAT model (same structure as before, 2 classes)
# ----------------------------
class GATTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=8,
                 num_classes=NUM_CLASSES, dropout=0.3):
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
def train_epoch(model, optimizer, criterion, loader, device, threshold=0.5):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_aug, edge_aug = data_augmentation(data.x, data.edge_index, training=True)
        out = model(x_aug, edge_aug)        # (N, 2)
        loss = criterion(out, data.y)       # y: (N,)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        # Apply custom threshold for prediction
        probs = F.softmax(out, dim=1)
        pred = (probs[:, 1] >= threshold).long()  # Predict "Has Text" if prob >= threshold
        total_correct += (pred == data.y).sum().item()
        total_samples += data.y.size(0)

    avg_loss = total_loss / len(loader)
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, acc


def evaluate(model, criterion, loader, device, threshold=0.5):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            # Apply custom threshold for prediction
            probs = F.softmax(out, dim=1)
            pred = (probs[:, 1] >= threshold).long()  # Predict "Has Text" if prob >= threshold
            total_correct += (pred == data.y).sum().item()
            total_samples += data.y.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return acc, avg_loss, np.array(all_preds), np.array(all_labels)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    input_dir = "./3.InputML"
    output_dir = "./5.OutputML_GAT_TEXT_HAS_HEURISTIC"
    os.makedirs(output_dir, exist_ok=True)

    # Load graphs
    graphml_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir)
         if f.endswith(".graphml")]
    )
    if len(graphml_files) < 2:
        raise ValueError("Need at least 2 graphml files for cross validation.")

    data_list = []
    for fp in graphml_files:
        print(f"\n=== Parsing {fp} ===")
        d = parse_graphml_to_pyg(fp)
        if d is not None:
            data_list.append(d)

    if len(data_list) < 2:
        raise ValueError("Parsed fewer than 2 valid graphs.")

    total_nodes = sum(d.y.size(0) for d in data_list)
    total_has_text = sum((d.y == 1).sum().item() for d in data_list)
    print(f"\nDataset summary:")
    print(f"  total nodes        : {total_nodes}")
    print(f"  total HasText nodes: {total_has_text}")
    print(f"  ratio HasText      : {total_has_text / max(total_nodes,1):.4f}")

    input_dim = data_list[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Cross-validation
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED) \
        if len(data_list) >= 5 else LeaveOneOut()
    cv_name = "5-fold" if isinstance(splitter, KFold) else "leave-one-out"
    print(f"Cross-validation: {cv_name}")

    fold_accuracies, fold_losses = [], []
    all_y_true, all_y_pred = [], []
    best_fold_acc, best_state = -1, None
    
    # Decision threshold: Higher = requires more confidence to predict "Has Text"
    # 0.5 = default, 0.6-0.7 = favor "No Text" predictions
    DECISION_THRESHOLD = 0.5614657

    def oversample_minority_class(train_data, target_ratio=0.2):
        """
        Oversample graphs with high proportion of 'Has Text' nodes to balance classes.
        target_ratio: aim for this proportion of positive samples in training
        """
        augmented = []
        for d in train_data:
            num_text = (d.y == 1).sum().item()
            num_total = d.y.size(0)
            ratio = num_text / max(num_total, 1)
            
            augmented.append(d)
            
            # Only duplicate graphs with very high text density
            if ratio > 0.30:  # Changed from 0.2 - more selective
                copies = 1 
                # copies = min(int(target_ratio / max(ratio, 0.05)), 2)  # Max 2 copies
                for _ in range(copies):
                    augmented.append(d)
        
        return augmented

    for fold, (train_idx, test_idx) in enumerate(splitter.split(data_list), start=1):
        print(f"\n==== Fold {fold} ====")
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_data = [data_list[i] for i in train_idx]
        train_data = oversample_minority_class(train_data, target_ratio=0.18)
        test_data = [data_list[i] for i in test_idx]
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        # Class distribution
        train_labels_all = torch.cat([d.y for d in train_data], dim=0)
        class_counts = torch.bincount(train_labels_all, minlength=NUM_CLASSES).float()
        print(f"Train class counts [No Text, Has Text]: {class_counts.tolist()}")

        # Instantiate model & optimizer
        # Instantiate model & optimizer
        model = GATTextModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_heads=8,
            num_classes=NUM_CLASSES,
            dropout=0.35,
        ).to(device)

        # Calculate class weights to handle imbalance
        total_samples = class_counts.sum()
        class_weights = total_samples / (NUM_CLASSES * class_counts)
        class_weights = torch.clamp(class_weights, min=1.0, max=12)  # Cap at 5x - balanced approach
        print(f"Class weights [No Text, Has Text]: {class_weights.tolist()}")
        
        criterion = FocalLoss(alpha=1, gamma=2.2, class_weights=class_weights.to(device))  # Lower alpha and gamma
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=40)

        best_val_acc = 0.0
        patience_ctr, patience = 0, 120
        epochs = 400
        train_losses, test_losses, test_accuracies, lrs = [], [], [], []

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_epoch(model, optimizer, criterion, train_loader, device, DECISION_THRESHOLD)
            te_acc, te_loss, te_preds, te_labels = evaluate(model, criterion, test_loader, device, DECISION_THRESHOLD)

            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            test_accuracies.append(te_acc)
            lrs.append(optimizer.param_groups[0]["lr"])

            scheduler.step(te_loss)

            if epoch % 10 == 0 or epoch <= 10:
                if len(set(te_labels.tolist())) > 1:
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

            # early stopping
            if te_acc > best_val_acc:
                best_val_acc = te_acc
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # intermediate plots
            if plot_training_validation and epoch % 25 == 0:
                plot_training_validation(train_losses, test_losses,
                                         os.path.join(fold_dir, "training_validation.png"))
                plot_validation_accuracy(test_accuracies,
                                         os.path.join(fold_dir, "validation_accuracy.png"))
                plot_lr_schedule(lrs, os.path.join(fold_dir, "lr_schedule.png"))

        # Final eval for this fold
        te_acc_final, te_loss_final, y_pred_fold, y_true_fold = evaluate(
            model, criterion, test_loader, device, DECISION_THRESHOLD
        )
        fold_accuracies.append(te_acc_final)
        fold_losses.append(te_loss_final)
        all_y_true.extend(y_true_fold.tolist())
        all_y_pred.extend(y_pred_fold.tolist())

        # Confusion matrix per fold
        if vg_plot_confusion_matrix:
            vg_plot_confusion_matrix(
                np.array(y_true_fold),
                np.array(y_pred_fold),
                LABEL_NAMES,
                os.path.join(fold_dir, "confusion_matrix.png"),
            )
        else:
            cm = confusion_matrix(y_true_fold, y_pred_fold, labels=range(NUM_CLASSES))
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
            plt.title(f"Fold {fold} Confusion Matrix (Text Has/No)")
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"))
            plt.close()

        if plot_class_accuracies:
            plot_class_accuracies(
                np.array(y_true_fold),
                np.array(y_pred_fold),
                LABEL_NAMES,
                os.path.join(fold_dir, "class_accuracies_histogram.png"),
            )

        if te_acc_final > best_fold_acc:
            best_fold_acc = te_acc_final
            best_state = model.state_dict()

        print(f"Fold {fold} Final Test Accuracy: {te_acc_final:.4f}")

    # ---- CV summary ----
    fold_accuracies = np.array(fold_accuracies)
    print("\n==== TEXT HAS/NO (HEURISTIC) CV SUMMARY ====")
    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {i}: Acc={acc:.4f}")
    print(f"Mean Acc: {fold_accuracies.mean():.4f} | Std: {fold_accuracies.std():.4f}")

    # ---- Save best model ----
    best_model = GATTextModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_heads=8,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    ).to(device)
    best_model.load_state_dict(best_state)

    pth_path = os.path.join(output_dir, "trained_model_text_has_heuristic.pth")
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_dim": input_dim,
            "num_classes": NUM_CLASSES,
            "hidden_dim": 128,
            "heads": 8,
            "dropout": 0.3,
            "best_fold_acc": best_fold_acc,
            "cv_mean_acc": fold_accuracies.mean(),
            "cv_std_acc": fold_accuracies.std(),
            "seed": SEED,
            "cv_mode": cv_name,
            "label_names": LABEL_NAMES,
            "task_type": "binary_text_has_no_heuristic",
        },
        pth_path,
    )
    print("Saved best PyTorch weights to:", pth_path)

    # ---- ONNX export ----
    onnx_path = os.path.join(output_dir, "trained_model_text_has_heuristic.onnx")
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

    # ---- Overall metrics ----
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Per-class metrics for [No Text, Has Text]
    rows = []
    for c_idx, name in enumerate(LABEL_NAMES):
        y_true_c = (all_y_true == c_idx).astype(int)
        y_pred_c = (all_y_pred == c_idx).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_c,
            y_pred_c,
            average="binary",
            zero_division=0,
        )
        rows.append({"Class": name, "Precision": prec, "Recall": rec, "F1-Score": f1})

    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(output_dir, "per_class_metrics_text_has_heuristic.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print("Per-class metrics saved to:", metrics_csv)
    print(metrics_df)

    # Overall confusion matrix
    if vg_plot_confusion_matrix:
        vg_plot_confusion_matrix(
            all_y_true,
            all_y_pred,
            LABEL_NAMES,
            os.path.join(output_dir, "confusion_matrix_overall_text_has_heuristic.png"),
        )
    else:
        cm = confusion_matrix(all_y_true, all_y_pred, labels=range(NUM_CLASSES))
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
        )
        plt.title("Overall Confusion Matrix (Text Has/No, Heuristic)")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_overall_text_has_heuristic.png"))
        plt.close()

    if plot_class_accuracies:
        plot_class_accuracies(
            all_y_true,
            all_y_pred,
            LABEL_NAMES,
            os.path.join(output_dir, "class_accuracies_histogram_overall_text_has_heuristic.png"),
        )

