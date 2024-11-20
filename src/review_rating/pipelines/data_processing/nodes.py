import mlflow
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer

from .dataset import ReviewDataset


def split_data(df, parameters):
    """
        Split data into training and test sets.
        Args:
            df: Pandas DataFrame
            parameters: Pipeline parameters
        Returns:
            train_df: Pandas DataFrame
            test_df: Pandas DataFrame
    """
    df = df.head(5000) # To train on a subset of data for pipeline testing purposes
    mlflow.log_input(
        mlflow.data.pandas_dataset.from_pandas(df, source="raw_data"),
        context="raw_data",
    )

    train_df, test_df = train_test_split(
        df,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=df["Score"],
    )

    mlflow.log_input(
        mlflow.data.pandas_dataset.from_pandas(train_df, source="train"),
        context="training",
    )
    mlflow.log_input(
        mlflow.data.pandas_dataset.from_pandas(test_df, source="test"),
        context="validation",
    )

    return train_df, test_df


def prepare_datasets(train_df, test_df):
    """
        Prepare PyTorch datasets.
        Args:
            train_df: Pandas DataFrame
            test_df: Pandas DataFrame
        Returns:
            train_dataset: PyTorch dataset
            test_dataset: PyTorch dataset
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = ReviewDataset(
        train_df["Text"].values, train_df["Score"].values, tokenizer
    )
    test_dataset = ReviewDataset(
        test_df["Text"].values, test_df["Score"].values, tokenizer
    )

    return train_dataset, test_dataset, tokenizer
