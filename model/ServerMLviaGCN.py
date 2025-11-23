import os
import random
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from torch.nn import Linear
import networkx as nx
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visualize_graph import (
    plot_training_validation, plot_test_accuracy, plot_confusion_matrix,
    plot_prediction_distribution, plot_validation_accuracy, plot_lr_schedule,
    plot_class_accuracies
)

# sklearn for CV + precision/recall/F1
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold, LeaveOneOut


# ----------------------------
# STRONGER REPRODUCIBILITY
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
# NOTE: do NOT force torch.use_deterministic_algorithms(True)
# because it can crash on CUDA (CuBLAS nondeterminism).


# 2. GraphML Parsing Function
def parse_graphml_to_pyg(filepath):
    G = nx.read_graphml(filepath)

    ignore_attributes = {
        'guid', 'info_string', 'label_guid', 'marker_guid',
        'embedded_door_guid', 'embedded_in_wall_guid', 'zone_stamp_guid'
    }

    all_attributes = set()
    for _, node_data in G.nodes(data=True):
        all_attributes.update(node_data.keys())

    for _, node_data in G.nodes(data=True):
        for attr in all_attributes:
            if attr in ignore_attributes:
                if attr in node_data:
                    del node_data[attr]
            else:
                node_data.setdefault(attr, 0.0)
                try:
                    node_data[attr] = float(node_data[attr])
                except ValueError:
                    node_data[attr] = 0.0

    numerical_attributes = [
        attr for attr in all_attributes
        if attr not in ignore_attributes and attr != 'label_type'
    ]
    numerical_attributes.sort()

    element_types = set(node_data['element_type'] for _, node_data in G.nodes(data=True))
    element_type_dict = {element_type: idx for idx, element_type in enumerate(element_types)}

    for _, node_data in G.nodes(data=True):
        node_data['element_type'] = element_type_dict[node_data['element_type']]

        room_numbers = set(node_data['room_number'] for _, node_data in G.nodes(data=True))
        room_number_dict = {room: idx for idx, room in enumerate(room_numbers)}

        room_names = set(node_data['room_name'] for _, node_data in G.nodes(data=True))
        room_name_dict = {room: idx for idx, room in enumerate(room_names)}

        node_data['room_number'] = room_number_dict[node_data['room_number']]
        node_data['room_name'] = room_name_dict[node_data['room_name']]

    node_features = [
        [node_data.get(attr, 0.0) for attr in numerical_attributes] +
        [node_data['element_type'], node_data['room_number'], node_data['room_name']]
        for _, node_data in G.nodes(data=True)
    ]

    labels = [node_data['label_type'] for _, node_data in G.nodes(data=True) if 'label_type' in node_data]

    features_tensor = torch.tensor(node_features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    data = from_networkx(G)
    data.x = features_tensor
    data.y = labels_tensor

    return data


# 3. Model Definitions
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# 4. Training / Eval Functions
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)

        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 0.001 * l2_reg

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.y.size(0)
        total_samples += data.y.size(0)
    return total_loss / total_samples


def evaluate(model, criterion, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            total_correct += (pred == data.y).sum().item()
            total_samples += data.y.size(0)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.y.size(0)
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    # ONE combined directory
    input_dir = "/mnt/UbuntuSSD/Spatial GNN/model/3.InputML"

    all_file_paths = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".graphml")
    ])

    if len(all_file_paths) < 2:
        raise ValueError("Need at least 2 graphml files for cross validation.")

    print("ALL graphs:", all_file_paths)

    # Output folder
    output_folder = "/mnt/UbuntuSSD/Spatial GNN/model/4.OutputML_GCN"
    os.makedirs(output_folder, exist_ok=True)

    # Decide CV mode
    if len(all_file_paths) >= 5:
        splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_name = "5-fold"
    else:
        splitter = LeaveOneOut()
        cv_name = "leave-one-out"

    print(f"\n✅ Using {cv_name} cross validation with SEED={SEED}\n")

    # Load ALL graphs once
    all_data = [parse_graphml_to_pyg(fp) for fp in all_file_paths]

    # Dimensions from first graph
    input_dim = all_data[0].x.shape[1]
    output_dim = 5

    label_names = ['No Label', 'Wall with Dimension', 'Connected Label', 'Door Marker', 'Zone Stamp']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Using device:", device)

    fold_accuracies = []
    fold_losses = []

    all_y_true = []
    all_y_pred = []

    best_fold_acc = -1
    best_fold_state = None

    for fold, (train_idx, test_idx) in enumerate(splitter.split(all_data), start=1):
        print(f"\n================ FOLD {fold} ================")

        fold_dir = os.path.join(output_folder, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        # Class weights from TRAIN only
        all_train_labels = torch.cat([d.y for d in train_data], dim=0)
        class_counts = torch.bincount(all_train_labels, minlength=output_dim).float()
        class_weights = (class_counts.sum() / (output_dim * class_counts)).clamp(max=10.0)

        model = GCNModel(input_dim=input_dim, hidden_dim=128, output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=30)

        train_losses, test_losses, test_accuracies, lrs = [], [], [], []

        # --------------------
        # Training
        # --------------------
        for epoch in range(1, 751):
            tr_loss = train(model, optimizer, criterion, train_loader, device)
            train_losses.append(tr_loss)

            te_acc, te_loss = evaluate(model, criterion, test_loader, device)
            test_accuracies.append(te_acc)
            test_losses.append(te_loss)

            scheduler.step(te_loss)
            lrs.append(optimizer.param_groups[0]["lr"])

            print(
                f"Fold {fold} | Epoch {epoch:03d} "
                f"TrainLoss={tr_loss:.4f} TestAcc={te_acc:.4f} TestLoss={te_loss:.4f} "
                f"LR={lrs[-1]:.3g}"
            )

            # Plots per epoch (same as your flow)
            plot_training_validation(train_losses, test_losses, os.path.join(fold_dir, "training_validation_plot.png"))
            plot_validation_accuracy(test_accuracies, os.path.join(fold_dir, "validation_accuracy_plot.png"))
            plot_lr_schedule(lrs, os.path.join(fold_dir, "learning_rate_schedule.png"))

        # Final fold test accuracy
        final_acc = test_accuracies[-1]
        final_loss = test_losses[-1]
        fold_accuracies.append(final_acc)
        fold_losses.append(final_loss)

        print(f"\n✅ Fold {fold} Final Test Accuracy: {final_acc:.4f}")

        # Save best fold model
        if final_acc > best_fold_acc:
            best_fold_acc = final_acc
            best_fold_state = model.state_dict()

        # Collect predictions for overall metrics
        y_true_fold, y_pred_fold = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1).cpu().numpy()
                y_pred_fold.extend(pred)
                y_true_fold.extend(data.y.cpu().numpy())

        all_y_true.extend(y_true_fold)
        all_y_pred.extend(y_pred_fold)

        # Per-fold confusion matrix
        plot_confusion_matrix(
            np.array(y_true_fold), np.array(y_pred_fold), label_names,
            os.path.join(fold_dir, "confusion_matrix.png")
        )

        plot_class_accuracies(
            np.array(y_true_fold), np.array(y_pred_fold), label_names,
            os.path.join(fold_dir, "class_accuracies_histogram_GCN.png")
        )

    # ============================
    # OVERALL CV SUMMARY
    # ============================
    fold_accuracies = np.array(fold_accuracies)
    fold_losses = np.array(fold_losses)

    print("\n================ CV SUMMARY ================")
    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {i}: Acc={acc:.4f}")

    print("-------------------------------------------")
    print(f"Mean Accuracy: {fold_accuracies.mean():.4f}")
    print(f"Std  Accuracy: {fold_accuracies.std():.4f}")
    print("===========================================\n")

    # ============================
    # SAVE BEST FOLD MODEL
    # ============================
    best_model = GCNModel(input_dim=input_dim, hidden_dim=128, output_dim=output_dim).to(device)
    best_model.load_state_dict(best_fold_state)

    pth_path = os.path.join(output_folder, "trained_model.pth")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": 128,
        "best_fold_acc": best_fold_acc,
        "cv_mean_acc": fold_accuracies.mean(),
        "cv_std_acc": fold_accuracies.std(),
        "seed": SEED,
        "cv_mode": cv_name,
    }, pth_path)
    print("✅ Saved BEST PyTorch weights to:", pth_path)

    onnx_path = os.path.join(output_folder, "trained_model.onnx")
    dummy_x = torch.randn(1, input_dim).to(device)
    dummy_edge = torch.tensor([[0], [0]], dtype=torch.long).to(device)

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
            "logits": {0: "num_nodes"}
        }
    )
    print("✅ Saved BEST ONNX model to:", onnx_path)

    # ============================
    # OVERALL METRICS (ALL FOLDS)
    # ============================
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=range(output_dim), zero_division=0
    )

    metrics_df = pd.DataFrame({
        "Class": label_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    metrics_csv_path = os.path.join(output_folder, "per_class_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print("📊 Saved overall per-class precision/recall/F1 to:", metrics_csv_path)

    print("\n===== OVERALL PER-CLASS PRECISION / RECALL / F1 =====")
    print(metrics_df)
    print("=====================================================\n")
    print("Macro F1:", f1.mean())
    print("Weighted F1:", np.average(f1, weights=np.bincount(all_y_true)))
    print()

    # Overall confusion matrix
    plot_confusion_matrix(
        all_y_true, all_y_pred, label_names,
        os.path.join(output_folder, "confusion_matrix.png")
    )

    plot_class_accuracies(
        all_y_true, all_y_pred, label_names,
        os.path.join(output_folder, "class_accuracies_histogram_GCN.png")
    )
