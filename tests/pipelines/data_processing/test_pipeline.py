import pytest
import pandas as pd
import torch
from transformers import RobertaTokenizer
from unittest.mock import Mock, patch
from src.review_rating.pipelines.data_processing.nodes import split_data, prepare_datasets
from src.review_rating.pipelines.data_processing.dataset import ReviewDataset

@pytest.fixture
def sample_reviews_df():
    """
    Create a balanced sample dataset with multiple samples per class for stratification
    """
    reviews = []
    scores = []
    product_ids = []
    user_ids = []
    
    # Generate 4 reviews for each score (1-5)
    for score in range(1, 6):
        for i in range(4):
            reviews.append(f"Sample review {i+1} for score {score}")
            scores.append(score)
            product_ids.append(f"P{score}_{i}")
            user_ids.append(f"U{score}_{i}")
    
    return pd.DataFrame({
        'Text': reviews,
        'Score': scores,
        'ProductId': product_ids,
        'UserId': user_ids
    })

@pytest.fixture
def model_parameters():
    return {
        "test_size": 0.4,
        "random_state": 42
    }

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    def mock_encode(text, **kwargs):
        max_length = kwargs.get('max_length', 512)
        return {
            'input_ids': torch.tensor([[1] * max_length]),
            'attention_mask': torch.tensor([[1] * max_length])
        }
    tokenizer.__call__ = Mock(side_effect=mock_encode)
    return tokenizer

def test_split_data(sample_reviews_df, model_parameters):
    with patch('mlflow.log_input') as mock_log_input:
        # First verify we have enough samples per class
        value_counts = sample_reviews_df['Score'].value_counts()
        assert all(count >= 2 for count in value_counts.values), \
            "Each class must have at least 2 samples for stratification"
        
        train_df, test_df = split_data(sample_reviews_df, model_parameters)
        
        # Assert basic split properties
        assert len(train_df) + len(test_df) == len(sample_reviews_df)
        expected_test_size = int(len(sample_reviews_df) * model_parameters["test_size"])
        assert abs(len(test_df) - expected_test_size) <= 1  # Allow for rounding
        
        # Verify class distribution is maintained
        train_dist = train_df['Score'].value_counts().sort_index()
        test_dist = test_df['Score'].value_counts().sort_index()
        original_dist = sample_reviews_df['Score'].value_counts().sort_index()
        
        # Check if all classes are present in both splits
        assert set(train_df['Score'].unique()) == set(sample_reviews_df['Score'].unique())
        assert set(test_df['Score'].unique()) == set(sample_reviews_df['Score'].unique())
        
        # Verify mlflow logging was called
        assert mock_log_input.call_count == 3

def test_prepare_datasets(sample_reviews_df):
    train_df = sample_reviews_df.iloc[:10]
    test_df = sample_reviews_df.iloc[10:]
    
    with patch('transformers.RobertaTokenizer.from_pretrained') as mock_tokenizer_cls:
        mock_tokenizer = Mock(spec=RobertaTokenizer)
        def mock_encode(text, **kwargs):
            return {
                'input_ids': torch.randint(0, 1000, (1, kwargs.get('max_length', 512))),
                'attention_mask': torch.ones(1, kwargs.get('max_length', 512))
            }
        mock_tokenizer.__call__ = mock_encode
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        train_dataset, test_dataset, tokenizer = prepare_datasets(train_df, test_df)
        
        assert isinstance(train_dataset, ReviewDataset)
        assert isinstance(test_dataset, ReviewDataset)
        assert len(train_dataset) == len(train_df)
        assert len(test_dataset) == len(test_df)

class TestReviewDataset:
    @pytest.fixture
    def valid_dataset(self, mock_tokenizer):
        reviews = ['Great product!', 'Bad product!']
        scores = [5, 1]
        return ReviewDataset(reviews, scores, mock_tokenizer)
    
    def test_valid_dataset(self, valid_dataset):
        assert len(valid_dataset) == 2
        item = valid_dataset[0]
        assert all(k in item for k in ['input_ids', 'attention_mask', 'labels'])
        assert isinstance(item['labels'], torch.Tensor)
        assert item['labels'].item() == 4  # 5-1 for zero-based indexing
    
    def test_empty_review(self, mock_tokenizer):
        reviews = ['']
        scores = [5]
        dataset = ReviewDataset(reviews, scores, mock_tokenizer)
        item = dataset[0]
        assert all(k in item for k in ['input_ids', 'attention_mask', 'labels'])
    
    def test_max_length_truncation(self, mock_tokenizer):
        long_review = ' '.join(['word'] * 1000)
        max_length = 128
        dataset = ReviewDataset([long_review], [5], mock_tokenizer, max_length=max_length)
        item = dataset[0]
        assert item['input_ids'].size(0) == max_length
        assert item['attention_mask'].size(0) == max_length
    
    def test_score_conversion(self, mock_tokenizer):
        """Test that scores are properly converted to zero-based indices"""
        reviews = ['Test review'] * 5
        scores = [1, 2, 3, 4, 5]  # 1-based scores
        dataset = ReviewDataset(reviews, scores, mock_tokenizer)
        
        for i, expected_label in enumerate(range(5)):  # 0-based labels
            item = dataset[i]
            assert item['labels'].item() == expected_label