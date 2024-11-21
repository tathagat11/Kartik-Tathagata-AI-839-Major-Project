import logging
from collections import OrderedDict

import matplotlib
import mlflow
import numpy as np
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator
from sklearn.metrics import classification_report
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from tqdm.auto import tqdm
# from weightwatcher import WeightWatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

matplotlib.use('Agg')

def compute_metrics(pred):
    """
    Compute basic metrics during training.
    Args:
        pred: Prediction results
    Returns:
        classification report
    """
    labels = pred.label_ids + 1
    preds = pred.predictions.argmax(-1) + 1
    return {"classification_report": classification_report(labels, preds)}

def train_model(train_dataset, test_dataset, params):
    """
    Train the model.
    Args:
        train_dataset: PyTorch dataset for training
        test_dataset: PyTorch dataset for validation
        params: Training parameters
    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)
    
    
    mlflow.log_params({
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "class_distribution": dict(enumerate(np.bincount([item["labels"].item() for item in train_dataset]).tolist()))
    })

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=5,
        problem_type="single_label_classification"
    )
        
    accelerator = Accelerator()
    model = accelerator.prepare(model)
        
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**params),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
        
    train_results = trainer.train()
    mlflow.log_metrics({"train_loss": train_results.training_loss})
        
    logger.info("Saving model...")
    try:
        # Properly unwrap and prepare model for saving
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model = unwrapped_model.cpu()

        # Save the unwrapped model's state dict
        state_dict = unwrapped_model.state_dict()
            
        # Create a fresh model instance for saving
        clean_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=5,
            problem_type="single_label_classification"
        )
        clean_model.load_state_dict(state_dict)
            
        mlflow.pytorch.log_model(
            clean_model,
            "model",
            registered_model_name="review_rating_model",
        )
            
        logger.info(f"Model URI: {mlflow.get_artifact_uri('model')}")
            
    except Exception as e:
        logger.error(f"Model saving failed: {str(e)}", exc_info=True)
        raise
            
    return clean_model


def evaluate_model(model, test_dataset):
    """
    Evaluate the model.
    Args:
        model: Trained model
        test_dataset: PyTorch dataset for evaluation
    Returns:
        classification report
    """
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model if needed
    if isinstance(model, (dict, OrderedDict)):
        eval_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=5,
            problem_type="single_label_classification"
        )
        eval_model.load_state_dict(model)
        model = eval_model

    model = model.to(device)
    model.eval()
    
    # Batch evaluation for better performance
    batch_size = 128
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(preds + 1)
            true_labels.extend(batch["labels"].cpu().numpy() + 1)

    report = classification_report(true_labels, predictions)
    metrics = classification_report(true_labels, predictions, output_dict=True)

    # Log metrics
    mlflow.log_text(report, "classification_report.txt")
    try:
        mlflow.log_metrics({
            f"{k}_f1-score": v["f1-score"]
            for k, v in metrics.items() 
            if isinstance(v, dict)
        })
    except Exception as e:
        logger.error(f"Failed to log metrics: {str(e)}", exc_info=True)

    return report
