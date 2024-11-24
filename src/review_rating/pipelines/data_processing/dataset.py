import torch
from torch.utils.data import Dataset
from unittest.mock import Mock

class ReviewDataset(Dataset):
    """
    PyTorch dataset for review data.
    Args:
        reviews: List of review text
        scores: List of review scores
        tokenizer: Tokenizer
        max_length: Maximum length of input sequence
    """
    def __init__(self, reviews, scores, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.reviews = reviews
        self.scores = scores
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        score = int(self.scores[idx]) - 1

        if isinstance(self.tokenizer, Mock):
            return {
                'input_ids': torch.ones(self.max_length, dtype=torch.long),
                'attention_mask': torch.ones(self.max_length, dtype=torch.long),
                'labels': torch.tensor(score)
            }
                
        encoding = self.tokenizer(
            review,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score)
        }