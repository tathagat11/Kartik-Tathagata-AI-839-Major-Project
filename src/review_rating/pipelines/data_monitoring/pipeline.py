from kedro.pipeline import Pipeline, node
from .nodes import run_data_validation

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_data_validation,
                inputs=[
                    "train_data",
                    "test_data",
                    "params:monitoring",
                ],
                outputs="monitoring_results",
                name="data_validation_node",
            ),
        ]
    )