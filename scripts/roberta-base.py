import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

import torch
torch.cuda.empty_cache() 

# Custom dataset class
class ReviewDataset(Dataset):
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

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Convert back to 1-5 range for reporting
    labels = labels + 1
    preds = preds + 1
    
    return {'classification_report': classification_report(labels, preds)}

def main():
    # Load data
    df = pd.read_csv("data/01_raw/Reviews.csv")
    
    # Split data
    train_texts, val_texts, train_scores, val_scores = train_test_split(
        df['Text'].values, 
        df['Score'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['Score'].values
    )

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=5,
        problem_type="single_label_classification"
    )

    # Create datasets
    train_dataset = ReviewDataset(train_texts, train_scores, tokenizer)
    val_dataset = ReviewDataset(val_texts, val_scores, tokenizer)

    # Calculate class weights for handling imbalance
    labels = df['Score'].values - 1  # Convert to 0-4 range
    class_counts = np.bincount(labels)
    total = len(labels)
    class_weights = torch.FloatTensor([total / count for count in class_counts])

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="data/08_reporting/results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="data/09_logs/roberta_base_logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=2e-5,
    )

    training_args = TrainingArguments(
        output_dir="data/08_reporting/results",
        num_train_epochs=1,              # One epoch should be sufficient for fine-tuning
        per_device_train_batch_size=16,  # Conservative batch size for 6GB GPU
        per_device_eval_batch_size=16,   # Same as training batch size
        warmup_ratio=0.1,               # 10% of total steps for warmup
        weight_decay=0.01,
        logging_dir="data/09_logs/roberta_base_logs",
        logging_steps=50,               # More frequent logging
        evaluation_strategy="steps",
        eval_steps=100,                 # Evaluate every 100 steps
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=2e-5,
        fp16=True,                      # Mixed precision training - crucial for memory savings
        gradient_accumulation_steps=2,   # Effective batch size of 32
        max_grad_norm=1.0,              # Gradient clipping to prevent instability
        dataloader_num_workers=2        # Don't overload system memory
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained("data/07_models/roberta_model/review_classifier_model")
    tokenizer.save_pretrained("data/07_models/roberta_model/review_classifier_model")

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print("\nValidation Results:")
    print(eval_results)

if __name__ == "__main__":
    main()