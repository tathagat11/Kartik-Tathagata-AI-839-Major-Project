import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from transformers import RobertaForSequenceClassification
from src.review_rating.pipelines.data_processing.dataset import ReviewDataset
from src.review_rating.pipelines.data_science.nodes import train_model, evaluate_model, compute_metrics

@pytest.fixture
def mock_datasets():
    train_dataset = Mock()
    test_dataset = Mock()
    
    # Add __len__ method
    train_dataset.__len__ = Mock(return_value=10)
    test_dataset.__len__ = Mock(return_value=5)
    
    # Add __getitem__ method with dummy data
    dummy_item = {
        'input_ids': torch.ones(128, dtype=torch.long),
        'attention_mask': torch.ones(128, dtype=torch.long),
        'labels': torch.tensor(2)
    }
    train_dataset.__getitem__ = Mock(return_value=dummy_item)
    test_dataset.__getitem__ = Mock(return_value=dummy_item)
    
    return train_dataset, test_dataset

@pytest.fixture
def training_params():
    return {
        "output_dir": ".test_data/tmp_test_model",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "warmup_steps": 2,
        "learning_rate": 2e-5,
        "logging_dir": "./tmp_test_logs",
        "logging_steps": 1,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss"
    }

def test_compute_metrics():
    class PredictionOutput:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids
    
    # Create dummy predictions and labels
    predictions = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])  # 2 samples, 3 classes
    labels = np.array([2, 0])  # True labels (zero-based)
    
    pred_output = PredictionOutput(predictions, labels)
    metrics = compute_metrics(pred_output)
    
    assert isinstance(metrics, dict)
    assert "classification_report" in metrics

@pytest.fixture
def mock_dataset():
    dataset = Mock()
    dataset.__len__ = Mock(return_value=2)
    dataset.__getitem__ = Mock(return_value={
        'input_ids': torch.ones(4, dtype=torch.long),
        'attention_mask': torch.ones(4, dtype=torch.long),
        'labels': torch.tensor(1, dtype=torch.long)
    })
    return dataset

def test_evaluate_model(mock_dataset):
    mock_model = Mock()
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    mock_outputs = Mock()
    mock_outputs.logits = torch.tensor([[0.1, 0.9]] * 2)
    mock_model.return_value = mock_outputs

    with patch('torch.utils.data.DataLoader', return_value=[{
            'input_ids': torch.ones(4, dtype=torch.long),
            'attention_mask': torch.ones(4, dtype=torch.long),
            'labels': torch.tensor([1] * 4, dtype=torch.long)
        }]), \
         patch('mlflow.log_text'), \
         patch('mlflow.log_metrics'):
        report = evaluate_model(mock_model, mock_dataset)
        assert isinstance(report, str)