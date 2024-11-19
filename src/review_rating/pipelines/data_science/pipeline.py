from kedro.pipeline import Pipeline, node
from .nodes import train_model, evaluate_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=["train_dataset", "test_dataset", "params:model_params"],
                outputs="model",
                name="train_model_node"
            ),
            node(
                func=evaluate_model,
                inputs=["model", "test_dataset"],
                outputs="evaluation_report",
                name="evaluate_model_node"
            )
        ]
    )