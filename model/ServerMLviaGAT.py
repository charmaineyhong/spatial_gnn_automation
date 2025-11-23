import os
import random
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
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
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
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


# 2. GraphML Parsing Function with Annotation Labels
def parse_graphml_to_pyg(filepath):
    G = nx.read_graphml(filepath)
    
    # Get corresponding annotation CSV file
    base_filename = os.path.basename(filepath).replace('.graphml', '').replace('PPVC_', 'PPVC ')
    
    # Map different naming conventions
    name_mapping = {
        'PPVC 01': 'PPVC 01, 20_Typ',
        'PPVC 02': 'PPVC 02_Typ 1',
        'PPVC 04': 'PPVC 04_Typ 1', 
        'PPVC 05': 'PPVC 05, 16_Typ',
        'PPVC 06': 'PPVC 06_Typ',
        'PPVC 21': 'PPVC 21,31_Typ',
        'PPVC 22': 'PPVC 22_Typ'
    }
    
    csv_folder_name = name_mapping.get(base_filename, base_filename)
    annotation_path = f"/mnt/UbuntuSSD/Spatial GNN/Extracted Data/{csv_folder_name}/annotation.csv"
    
    # Load annotation data to determine which elements need annotations
    annotated_elements = set()
    if os.path.exists(annotation_path):
        try:
            import pandas as pd
            annotations_df = pd.read_csv(annotation_path)
            for _, row in annotations_df.iterrows():
                target_ids = str(row['target_element_ids']).split('|')
                for element_id in target_ids:
                    if element_id.strip() and element_id.strip() != 'nan':
                        annotated_elements.add(int(element_id.strip()))
        except Exception as e:
            print(f"Warning: Could not load annotations for {filepath}: {e}")

    ignore_attributes = {
        'guid', 'info_string', 'label_guid', 'marker_guid',
        'embedded_door_guid', 'embedded_in_wall_guid', 'zone_stamp_guid'
    }

    all_attributes = set()
    for _, node_data in G.nodes(data=True):
        all_attributes.update(node_data.keys())

    for _, node_data in G.nodes(data=True):
        for attr in list(all_attributes):
            if attr in ignore_attributes:
                if attr in node_data:
                    del node_data[attr]
            else:
                node_data.setdefault(attr, 0.0)
                try:
                    node_data[attr] = float(node_data[attr])
                except (ValueError, TypeError):
                    node_data[attr] = 0.0

    numerical_attributes = [
        attr for attr in all_attributes
        if attr not in ignore_attributes and attr not in ['label_type', 'category']
    ]
    numerical_attributes.sort()

    element_types = sorted(set(d.get("element_type", "none") for _, d in G.nodes(data=True)))
    element_type_dict = {t: i for i, t in enumerate(element_types)}

    room_numbers = sorted(set(d.get("room_number", "none") for _, d in G.nodes(data=True)))
    room_number_dict = {t: i for i, t in enumerate(room_numbers)}

    room_names = sorted(set(d.get("room_name", "none") for _, d in G.nodes(data=True)))
    room_name_dict = {t: i for i, t in enumerate(room_names)}

    # Encode categorical variables
    categories = sorted(set(d.get("category", "none") for _, d in G.nodes(data=True)))
    category_dict = {t: i for i, t in enumerate(categories)}

    for _, d in G.nodes(data=True):
        d["element_type"] = element_type_dict.get(d.get("element_type", "none"), 0)
        d["room_number"]  = room_number_dict.get(d.get("room_number", "none"), 0)
        d["room_name"]    = room_name_dict.get(d.get("room_name", "none"), 0)
        d["category_encoded"] = category_dict.get(d.get("category", "none"), 0)

    node_features = []
    labels = []
    
    for node_id, node_data in G.nodes(data=True):
        # Enhanced feature vector
        features = [node_data.get(attr, 0.0) for attr in numerical_attributes]
        features += [
            node_data.get('element_type', 0),
            node_data.get('room_number', 0), 
            node_data.get('room_name', 0),
            node_data.get('category_encoded', 0)
        ]
        
        node_features.append(features)
        
        # Binary annotation label: 1 if element needs annotation, 0 otherwise
        try:
            node_int_id = int(node_data.get('node_id', node_id))
            label = 1 if node_int_id in annotated_elements else 0
        except:
            label = 0
            
        labels.append(label)

    features_tensor = torch.tensor(node_features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    data = from_networkx(G)
    data.x = features_tensor
    data.y = labels_tensor

    return data


# 3. Enhanced GAT Model for Small Dataset
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        
        # Multi-layer GAT with residual connections
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim//2, heads=heads//2, dropout=dropout)
        
        # Feature normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * heads)
        self.bn3 = torch.nn.BatchNorm1d((hidden_dim//2) * (heads//2))
        
        # Classification head with dropout
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear((hidden_dim//2) * (heads//2), hidden_dim//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim//4, output_dim)
        )
        
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        # First GAT layer with residual connection
        x1 = F.elu(self.conv1(x, edge_index))
        if batch is not None:
            x1 = self.bn1(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Second GAT layer with residual
        x2 = F.elu(self.conv2(x1, edge_index))
        if batch is not None:
            x2 = self.bn2(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # Third GAT layer
        x3 = F.elu(self.conv3(x2, edge_index))
        if batch is not None:
            x3 = self.bn3(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        
        # Classification
        out = self.classifier(x3)
        return out


# 4. Enhanced Training with Data Augmentation for Small Datasets
def train(model, optimizer, criterion, train_loader, device, epoch, warmup_epochs=10):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        
        # L2 regularization (stronger for small datasets)
        l2_reg = sum(torch.norm(p)**2 for p in model.parameters())
        loss += 0.01 * l2_reg  # Increased from 0.001
        
        # Label smoothing for binary classification
        if epoch > warmup_epochs:
            # Soft targets to prevent overconfidence
            soft_targets = data.y.float() * 0.9 + 0.05
            loss_smooth = F.binary_cross_entropy_with_logits(out[:, 1], soft_targets)
            loss = 0.7 * loss + 0.3 * loss_smooth
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        total_correct += (pred == data.y).sum().item()
        total_samples += data.y.size(0)
    
    return total_loss / len(train_loader), total_correct / total_samples


def evaluate(model, criterion, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            
            total_correct += (pred == data.y).sum().item()
            total_samples += data.y.size(0)
            total_loss += criterion(out, data.y).item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(loader)
    
    return accuracy, avg_loss, all_preds, all_labels


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
    output_folder = "/mnt/UbuntuSSD/Spatial GNN/model/5.OutputML_GAT"
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

    # Print dataset statistics
    total_nodes = sum(len(data.y) for data in all_data)
    total_annotated = sum((data.y == 1).sum().item() for data in all_data)
    print(f"📊 Dataset Statistics:")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Annotated nodes: {total_annotated}")
    print(f"   Non-annotated nodes: {total_nodes - total_annotated}")
    print(f"   Annotation ratio: {total_annotated/total_nodes:.3f}")

    # Dimensions from first graph
    input_dim = all_data[0].x.shape[1]
    output_dim = 2  # Binary classification: 0=no annotation, 1=needs annotation

    label_names = ['No Annotation Needed', 'Needs Annotation']

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

        # Enhanced data loaders for small datasets
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        # Calculate class weights from TRAIN data only (important for imbalanced data)
        all_train_labels = torch.cat([d.y for d in train_data], dim=0)
        class_counts = torch.bincount(all_train_labels, minlength=output_dim).float()
        
        # Prevent division by zero and extreme weights
        class_counts = torch.clamp(class_counts, min=1.0)
        total_samples = class_counts.sum()
        class_weights = total_samples / (output_dim * class_counts)
        class_weights = torch.clamp(class_weights, max=10.0)  # Cap maximum weight
        
        print(f"   Class distribution: {class_counts.int().tolist()}")
        print(f"   Class weights: {class_weights.tolist()}")

        # Enhanced model with more capacity
        model = GATModel(
            input_dim=input_dim, 
            hidden_dim=128,  # Increased from 64
            output_dim=output_dim, 
            heads=4,  # More attention heads
            dropout=0.2  # Less aggressive dropout for small dataset
        ).to(device)
        
        # Optimized learning setup for small datasets
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # More patient scheduler for small datasets
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50
        )
        
        # Early stopping parameters
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 100  # More patience for small datasets

        train_losses, test_losses, test_accuracies, train_accuracies, lrs = [], [], [], [], []

        # --------------------
        # Enhanced Training Loop
        # --------------------
        epochs = 500  # Reduced from 750 for faster iteration with small dataset
        
        for epoch in range(1, epochs + 1):
            # Training with enhanced techniques
            tr_loss, tr_acc = train(model, optimizer, criterion, train_loader, device, epoch)
            train_losses.append(tr_loss)
            train_accuracies.append(tr_acc)

            # Validation
            te_acc, te_loss, te_preds, te_labels = evaluate(model, criterion, test_loader, device)
            test_accuracies.append(te_acc)
            test_losses.append(te_loss)

            scheduler.step(te_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)

            # Enhanced logging with more metrics
            if epoch % 10 == 0 or epoch <= 10:
                # Calculate precision/recall for binary classification
                if len(set(te_labels)) > 1:  # Avoid division by zero
                    precision = precision_score(te_labels, te_preds, zero_division=0)
                    recall = recall_score(te_labels, te_preds, zero_division=0)
                    f1 = f1_score(te_labels, te_preds, zero_division=0)
                else:
                    precision = recall = f1 = 0.0
                
                print(
                    f"Fold {fold} | Epoch {epoch:03d} | "
                    f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.4f} | "
                    f"TestLoss={te_loss:.4f} TestAcc={te_acc:.4f} | "
                    f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
                    f"LR={current_lr:.6f}"
                )

            # Early stopping check
            if te_acc > best_val_acc:
                best_val_acc = te_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Generate plots every 25 epochs for monitoring
            if epoch % 25 == 0:
                plot_training_validation(train_losses, test_losses,
                                         os.path.join(fold_dir, "training_validation_plot.png"))
                plot_validation_accuracy(test_accuracies,
                                         os.path.join(fold_dir, "validation_accuracy_plot.png"))
                plot_lr_schedule(lrs,
                                 os.path.join(fold_dir, "learning_rate_schedule.png"))

        # Collect final predictions for overall metrics
        te_acc_final, te_loss_final, y_pred_fold, y_true_fold = evaluate(model, criterion, test_loader, device)
        
        final_acc = te_acc_final
        final_loss = te_loss_final
        fold_accuracies.append(final_acc)
        fold_losses.append(final_loss)

        print(f"\n✅ Fold {fold} Final Test Accuracy: {final_acc:.4f}")

        # Save best fold model
        if final_acc > best_fold_acc:
            best_fold_acc = final_acc
            best_fold_state = model.state_dict()

        all_y_true.extend(y_true_fold)
        all_y_pred.extend(y_pred_fold)

        # Per-fold confusion matrix
        plot_confusion_matrix(
            np.array(y_true_fold), np.array(y_pred_fold), label_names,
            os.path.join(fold_dir, "confusion_matrix.png")
        )

        plot_class_accuracies(
            np.array(y_true_fold), np.array(y_pred_fold), label_names,
            os.path.join(fold_dir, "class_accuracies_histogram_GAT.png")
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
    best_model = GATModel(
        input_dim=input_dim, 
        hidden_dim=128, 
        output_dim=output_dim, 
        heads=4, 
        dropout=0.2
    ).to(device)
    best_model.load_state_dict(best_fold_state)

    pth_path = os.path.join(output_folder, "trained_model.pth")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": 128,
        "heads": 4,
        "dropout": 0.2,
        "best_fold_acc": best_fold_acc,
        "cv_mean_acc": fold_accuracies.mean(),
        "cv_std_acc": fold_accuracies.std(),
        "seed": SEED,
        "cv_mode": cv_name,
        "label_names": label_names,
        "task_type": "binary_annotation_prediction"
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
        os.path.join(output_folder, "class_accuracies_histogram_GAT.png")
    )
