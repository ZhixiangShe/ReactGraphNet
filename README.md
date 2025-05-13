# ReactGraphNet Framework for Molecular Reactivity Predcition

This repository provides a modular pipeline for graph neural network (GNN) based node classification using PyTorch Geometric. The framework is designed for predicting molecular reactivity based on molecular descriptors and reaction relationships in water treatments, focusing on reproducibility, clarity, and interpretability. The code includes capabilities for handling external generalization tests.

This codebase is associated with the research paper "Interpretable Graph Neural Network for Molecular Reactivity Prediction of Dissolved Organic Matter in Water Treatment".

## License

This software is released under the **MIT License**, an OSI-approved open source license. A copy of the license is included in the `LICENSE` file in this repository. This license permits the free use, modification, and distribution of the code under the terms specified, facilitating open research and development.

## Features

*   Data loading and preprocessing for external CSV datasets following a standardized format.
*   Support for multiple popular GNN architectures: GCN, GAT, GraphSAGE, TAGCN, and ClusterGCN.
*   Includes robust training, evaluation, and hyperparameter search capabilities.
*   Model explainability using GNNExplainer to understand important features and subgraphs.
*   Integrated pipeline for external generalization testing on new datasets.
*   Modular codebase with clear comments and docstrings to enhance readability and maintainability.

## Data Format

The code expects input data in CSV format, organized into separate files for node features, graph edges, and node labels. The required columns and format are as follows:

1.  **`features.csv`**: Contains node features.
    *   Must have a column named `node_id` for unique node identification.
    *   All other columns are treated as numerical features.
    *   Example:
        ```csv
        node_id,feature1,feature2,...
        mol_001_atom_1,0.1,1.5,...
        mol_001_atom_2,0.5,2.1,...
        ...
        ```

2.  **`edges.csv`**: Contains the graph's edge list.
    *   Must have columns named `source` and `target`, containing the `node_id`s of connected nodes.
    *   Example:
        ```csv
        source,target
        mol_001_atom_1,mol_001_atom_2
        mol_001_atom_1,mol_001_atom_3
        ...
        ```

3.  **`labels.csv`**: Contains the ground truth labels for nodes used in training and testing.
    *   Must have columns named `node_id` and `label`.
    *   Example:
        ```csv
        node_id,label
        mol_001_atom_1,removed
        mol_001_atom_2,produced
        ...
        ```

For **external prediction**, the code also expects `external_features.csv`, `external_edges.csv`, and potentially `external_labels.csv` (if ground truth is available for evaluation) following the same format.

## Directory Structure

<pre>
project_root/
├── data/                 # Input CSV files (features, edges, labels)
├── results/              # Saved models and outputs
├── src/                  # Source code modules
│   ├── data_utils.py
│   ├── models.py
│   ├── train_eval.py
│   ├── metrics_utils.py
│   ├── explainability.py
│   ├── external_predict.py
│   └── main_pipeline.py
├── requirements.txt
└── README.md
</pre>


## System Requirements

*   **Operating System:** Compatible with Windows, macOS, and Linux.
*   **Python Version:** Python 3.7 or higher is recommended.
*   **Software Dependencies:** All required Python packages are listed in the `requirements.txt` file. Key dependencies include:
    *   `numpy`
    *   `pandas`
    *   `torch` (PyTorch)
    *   `torch-geometric` (PyG)
    *   `scikit-learn`
    *   `matplotlib`
    *   `openpyxl` (for Excel output)
    *   `joblib` (for saving scaler and label encoder)
*   **Hardware:** A multi-core CPU is sufficient for basic execution. A GPU (especially NVIDIA with CUDA support) is strongly recommended to significantly accelerate model training and hyperparameter search, particularly on larger datasets.
*   **Tested Versions:** The code has been primarily developed and tested with Python 3.x and the package versions specified in `requirements.txt`. Users are advised to install dependencies from this file to ensure compatibility. Specific PyTorch and PyG versions depend on your CUDA/CPU setup; consult their installation guides for compatible versions.

## Installation Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ZhixiangShe/ReactGraphNet.git
    cd ReactGraphNet
    ```
2.  **Install dependencies:**
    *   **Install PyTorch:** Follow the official PyTorch installation guide ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) to install the version appropriate for your system (with or without CUDA).
    *   **Install PyTorch Geometric (PyG):** Follow the official PyG installation guide ([https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)). **It is critical to install PyG in a way that is compatible with your installed PyTorch version and CUDA setup.**
    *   **Install other requirements:** Once PyTorch and PyG are successfully installed, install the remaining libraries using pip:
        ```bash
        pip install -r requirements.txt
        ```
3.  **Typical Install Time:** After correctly installing PyTorch and PyG, the installation of other requirements from `requirements.txt` typically takes between 5 to 15 minutes on a standard internet connection. The PyTorch/PyG installation time can vary depending on network speed and required CUDA dependencies.

## Demo

This section explains how to run the code using a small sample dataset included in the repository for demonstration purposes.

1.  **Prepare Demo Data:** Ensure the sample data files (`features.csv`, `edges.csv`, `labels.csv`, and optionally external data files) are placed in the `data/new-2025/EO/GML/` directory within the cloned repository.
2.  **Run the Main Script:** Navigate to the `src/` directory and execute the `main_pipeline.py` script:
    ```bash
    cd src/
    python main_pipeline.py
    ```
3.  **Expected Output:** The script will print progress (e.g., epoch loss, accuracy) to the console during training and evaluation. Upon successful completion, it will generate several output files in the `results/` directory. Refer to the "Output Files" section below for a detailed list and description of these files.
4.  **Expected Run Time:** Running the complete pipeline with the provided demo dataset on a typical desktop CPU ([mention CPU specifics if known, e.g., i5/Ryzen 5 or better]) is expected to take approximately [**Estimate time here, e.g., 15-30 minutes**]. Using a GPU will significantly reduce this time.

## Instructions for Use (with Your Own Data)

To apply the ReactGraphNet framework to your own molecular reactivity data:

1.  **Prepare Your Data:** Format your node features, graph edges, and node labels into the three required CSV files (`features.csv`, `edges.csv`, `labels.csv`) as described in the "Data Format" section. If you intend to use the external prediction functionality, prepare `external_features.csv`, `external_edges.csv`, and optionally `external_labels.csv` in the same format.
2.  **Place Your Data:** Create the nested directory structure `data/new-2025/EO/GML/` within the repository's root, if it doesn't exist, and place your prepared CSV files inside this directory, overwriting or replacing the demo files.
3.  **Adjust Configuration (Optional):** You may need to adjust the `param_grid` or other configurations within the `main_pipeline.py` script or a dedicated configuration file (if implemented) to tune the model search to your specific dataset size and characteristics.
4.  **Run the Pipeline:** Navigate to the `src/` directory and run the main script:
    ```bash
    cd src/
    python main_pipeline.py
    ```
    The pipeline will load your data from the specified path, proceed with hyperparameter search, training, evaluation, and prediction on external data (if provided), saving all results to the `results/` directory.

## Output Files in results/ Directory

After running the `main_pipeline.py` script, the `results/` directory will be populated with the following output files:

*   `evaluation_metrics_best_params.xlsx`: Summarizes evaluation metrics (accuracy, precision, recall, F1-score, AUC) and the best hyperparameters found for each tested GNN model type on the internal train/test split.
*   `confusion_matrices.xlsx`: Contains confusion matrices for the best-performing model of each type on both the training and testing sets.
*   `roc_pr_curves.xlsx`: Stores data points (FPR, TPR, Precision, Recall) and AUC values necessary to plot ROC and Precision-Recall curves per class for each model on internal train/test sets.
*   `best_model_train_predictions.csv`: Lists predictions, true labels, and class probabilities for the training set using the overall best performing model identified during the hyperparameter search. Includes node IDs.
*   `best_model_test_predictions.csv`: Lists predictions, true labels, and class probabilities for the testing set using the overall best performing model. Includes node IDs.
*   `feature_importance.png`: PNG image visualizing the importance of different node features as determined by GNNExplainer for a selected example node.
*   `subgraph.pdf`: PDF file visualizing the computational subgraph surrounding a selected example node, highlighting important edges and nodes as identified by GNNExplainer.
*   `node_feature_importance.xlsx`: Detailed tabular data from the GNNExplainer analysis showing feature importance scores per node.
*   `subgraph_data.xlsx`: Detailed tabular data from the GNNExplainer analysis including edges in the subgraph and their importance scores.
*   `best_model.pth`: PyTorch state dictionary file containing the learned weights of the overall best performing GNN model.
*   `scaler.pkl`: Joblib file saving the state of the `StandardScaler` used for feature normalization.
*   `label_encoder.pkl`: Joblib file saving the state of the `LabelEncoder` used for transforming categorical labels into numerical format.
*   `external_evaluation_results.txt`: Text file containing evaluation metrics (if `external_labels.csv` was provided) for the predictions made on the external dataset.
*   `external_predictions.csv`: CSV file containing predictions, including node IDs, predicted labels, and class probabilities for the external dataset. If `external_labels.csv` was provided, it will also include the true labels.
*   `external_roc_pr.xlsx`: Excel file containing ROC and PR curve data for the external dataset if ground truth labels were available.
*   `external_roc_curves.png`: PNG image visualizing the ROC curves for each class on the external dataset (if ground truth was available).

## Contact

For questions, issues, or collaborations related to this codebase, please contact:

Zhixiang She
shezhixiang@ustc.edu.cn
