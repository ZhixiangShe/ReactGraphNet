# ReactGraphNet Framework for Molecular Reactivity Predcition

This repository provides a modular pipeline for graph neural network (GNN) based node classification using PyTorch Geometric. The code is organized for reproducibility and clarity, suitable for scientific publication and further research.

## Features

- Data loading and preprocessing for external CSV datasets  
- Multiple GNN architectures: GCN, GAT, GraphSAGE, TAGCN, and ClusterGCN  
- Robust training, evaluation, and hyperparameter search  
- Model explainability via GNNExplainer  
- External generalization test and result output  
- Modular codebase with clear comments and docstrings

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


## Usage

*   **`src/`:** Contains all Python source code modules, each responsible for a specific part of the pipeline.
*   **`data/`:**  Holds input CSV data files.  Ensure your data files (`features.csv`, `edges.csv`, `labels.csv`, `external_features.csv`, `external_edges.csv`, `external_labels.csv`) are placed in the `data/new-2025/EO/GML/` directory.
*   **`results/`:**  Output directory where all results (evaluation metrics, trained models, predictions, visualizations) will be saved. This directory will be created automatically upon running the main script.

## How to Run

1.  **Navigate to the `src/` directory:**

    ```bash
    cd src/
    ```

2.  **Run the main pipeline script:**

    ```bash
    python main_pipeline.py
    ```

    This script will:
    *   Load and preprocess the graph data.
    *   Perform hyperparameter search across different GNN models.
    *   Retrain the best performing model with optimal parameters.
    *   Save the best model, scaler, and label encoder.
    *   Generate explanations for a chosen node using GNNExplainer.
    *   Predict labels for external data using the best model and save the results along with evaluation metrics.
    *   Save all evaluation metrics, visualizations, and output data to the `results/` directory.

## Output Files in `results/` Directory

The `results/` directory will contain the following output files after running the pipeline:

*   **`evaluation_metrics_best_params.xlsx`:** Excel file containing evaluation metrics and best hyperparameters for each tested GNN model.
*   **`confusion_matrices.xlsx`:** Excel file containing confusion matrices for training and test sets for each model.
*   **`roc_pr_curves.xlsx`:** Excel file containing ROC and PR curve data for training and test sets for each model.
*   **`best_model_train_predictions.csv`:** CSV file with predictions on the training dataset using the best model.
*   **`best_model_test_predictions.csv`:** CSV file with predictions on the test dataset using the best model.
*   **`feature_importance.png`:** PNG image visualizing feature importance for a selected node, as explained by GNNExplainer.
*   **`subgraph.pdf`:** PDF file visualizing the subgraph important for the prediction of a selected node, as explained by GNNExplainer.
*   **`node_feature_importance.xlsx`:** Excel file containing detailed node feature importance data from GNNExplainer.
*   **`subgraph_data.xlsx`:** Excel file containing subgraph edge data and edge importance scores from GNNExplainer.
*   **`best_model.pth`:** PyTorch state dictionary file of the best trained GNN model.
*   **`scaler.pkl`:** Joblib file of the StandardScaler object used for feature scaling.
*   **`label_encoder.pkl`:** Joblib file of the LabelEncoder object used for label encoding.
*   **`external_evaluation_results.txt`:** Text file containing evaluation metrics for predictions on the external dataset.
*   **`external_predictions.csv`:** CSV file with predictions on the external dataset, including node IDs, true labels, predicted labels, and class probabilities.
*   **`external_roc_pr.xlsx`:** Excel file containing ROC and PR curve data for the external dataset.
*   **`external_roc_curves.png`:** PNG image of ROC curves for each class on the external dataset.

## Further Work and Enhancements

*   **Parameterize File Paths and Configurations:** Move hardcoded file paths and hyperparameter configurations to a configuration file (e.g., `.yaml` or `.json`) for easier modification and management.
*   **Implement Logging:** Add logging to track the progress of the pipeline and any errors encountered during execution.
*   **Advanced Hyperparameter Optimization:** Explore more advanced hyperparameter optimization techniques beyond grid search, such as Bayesian optimization or genetic algorithms.
*   **Experiment with More GNN Architectures and Layers:** Integrate other GNN models and layer types from PyTorch Geometric or custom implementations to potentially improve performance.
*   **Extend Explainability Methods:** Incorporate additional GNN explainability methods beyond GNNExplainer to gain a more comprehensive understanding of model behavior.
*   **Automated Documentation:** Set up automatic documentation generation (e.g., using Sphinx or MkDocs) to create comprehensive API documentation from docstrings.
*   **Unit and Integration Tests:** Implement unit and integration tests to ensure the robustness and correctness of the code.


---
**Author:** Zhixiang She
**Date:** [2025-04-21]
