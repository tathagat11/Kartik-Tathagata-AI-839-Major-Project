from kedro.pipeline import Pipeline, node

from .nodes import split_data, prepare_datasets

def create_pipeline(**kwargs) -> Pipeline:
    """
        Create the data processing pipeline.
        Args:
            kwargs: Pipeline parameters
        Returns:
            Pipeline
    """
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["reviews", "params:train_test_split"],
                outputs=["train_data", "test_data"],
                name="split_data_node",
            ),
            node(
                func=prepare_datasets,
                inputs=["train_data", "test_data"],
                outputs=["train_dataset", "test_dataset", "tokenizer"],
                name="prepare_datasets_node",
            ),
        ]
    )