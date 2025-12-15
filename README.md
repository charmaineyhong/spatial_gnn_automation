# Python GAT Model for Spatial Design Studio Revit Auto Annotator

This repository provides automation tools and models for spatial graph neural networks (GNNs), focusing on 4-class classification tasks using Graph Attention Networks (GAT). It includes Jupyter notebooks for experimentation, Python scripts for data inspection, inference, and batch processing, and utilities for handling spatial data from extracted sources.

## Features

- **Model Training**: GAT-based model for 4-class classification with focal loss, oversampling for minority classes, and cross-validation.
- **Inference**: Automated workflows for running predictions on CSV or GraphML inputs.
- **Data Inspection**: Scripts to analyze annotation data, including dimension and text umbrella extraction.
- **Batch Processing**: Tools for processing multiple files and saving results.
- **Visualization**: Optional plotting for training metrics, confusion matrices, and prediction distributions.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/charmaineyhong/spatial_gnn_automation.git
   cd spatial_gnn_automation
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install torch-geometric
   pip install pandas scikit-learn matplotlib seaborn
   ```
## User-Specific Configurations

- **Model Path**: The default model path in inference scripts (e.g., `automated_inference_workflow.py`) is set to `'./mostlatesttrained_model.pth'`. Update this to point to your trained model file, e.g., change to `'./path/to/your/trained_model.pth'`.
- **Batch File (`run_ppvc_inference.bat`)**: This batch file automates the inference process. You must update the following user-specific paths:
  - Change `call "C:\Users\charm\anaconda3\Scripts\activate.bat" base` to your Anaconda activation path, e.g., `call "C:\Users\charmaineyhong\anaconda3\Scripts\activate.bat" base`.
  - Change `cd /d "C:\Users\charm\Spatial GNN\model"` to the directory where your scripts are located, e.g., `cd /d "C:\Users\charmaineyhong\spatial_gnn_automation\model"`.
  - Change `--model "4trained_model.pth"` to your model's filename, e.g., `--model "mostlatesttrained_model.pth"`.
- **Data Paths**: In scripts like `inspect_annotation.py`, the base directory is set to `'../Extracted Data'`. Adjust this to your data folder, e.g., change to `'C:\Users\charmaineyhong\Documents\Extracted Data'`.
- **Google Drive Paths**: In Colab notebooks, paths like `'/content/drive/MyDrive/Spatial GNN'` are user-specific. Mount your Drive and update paths accordingly.

## Integrated Workflow with Revit

This repository works seamlessly with the [revit-addin-app](https://github.com/charmaineyhong/revit-addin-app) to automate spatial GNN predictions on Revit models:

1. **Export Data from Revit**: Use the Revit add-in to export model data. Open your Revit project, run the "PPVC Export" command, and select an output folder. This generates `nodes.csv`, `edges.csv`, and `annotation.csv` files containing the model's structural and annotation data.

2. **Run Inference with Python Model**: Use the exported CSVs as input to the GNN model. For example, run the automated inference workflow:
   ```bash
   python automated_inference_workflow.py --nodes_csv path/to/nodes.csv --edges_csv path/to/edges.csv --annotation_csv path/to/annotation.csv --out_csv predictions.csv
   ```
   Or use the batch file: `run_ppvc_inference.bat nodes.csv edges.csv predictions.csv`.

3. **Import Predictions Back to Revit** (Optional): Copy the `predictions.csv` to your Revit project directory. Use the "PPVC Auto Annotate" command in the Revit add-in to apply the predictions as annotations in the model.

This workflow enables automated labeling and annotation of spatial elements in BIM models using graph neural networks.

