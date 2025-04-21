"""
Model explainability (GNNExplainer, feature and subgraph visualization).
"""
import torch
from torch_geometric.explain import Explainer, GNNExplainer

def explain_node(model, data, node_index, out_dir):
    """
    Generates and saves node-level explanations.
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    explanation = explainer(data.x, data.edge_index, index=node_index)
    # Save feature importance and subgraph
    explanation.visualize_feature_importance(f"{out_dir}/feature_importance.png", top_k=17)
    explanation.visualize_graph(f"{out_dir}/subgraph.pdf")
    return explanation
