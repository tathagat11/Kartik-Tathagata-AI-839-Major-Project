import logging
from collections import OrderedDict

import matplotlib
import mlflow
import numpy as np
import sklearn
import torch
from accelerate import Accelerator
from sklearn.metrics import classification_report
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
# from weightwatcher import WeightWatcher

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
    Train the model with proper tracking and logging.
    Args:
        train_dataset: PyTorch dataset for training
        test_dataset: PyTorch dataset for testing/validation
        params: Training parameters
    Returns:
        trained model
    """
    # Clear GPU memory and log basic info
    torch.cuda.empty_cache()
    mlflow.log_param("train_size", len(train_dataset))
    mlflow.log_param("test_size", len(test_dataset))
    # Log class distribution
    logging.info("Loading raw scores...")
    raw_scores = [item["labels"].cpu().item() for item in train_dataset]
    class_counts = np.bincount(raw_scores)
    mlflow.log_param("class_distribution", dict(enumerate(class_counts.tolist())))

    # Initialize model and training
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=5, 
        problem_type="single_label_classification"
    )
    training_args = TrainingArguments(**params)
    accelerator = Accelerator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    # Train model
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed")

    # Prepare model for analysis
    unwrapped_model = accelerator.unwrap_model(model).cpu()

    # Analyze model weights
    logging.info("Analyzing model weights...")
    # watcher = WeightWatcher()
    # watcher_results = watcher.analyze(
    #     model=unwrapped_model,
    #     plot=True,
    #     savefig="data/08_reporting/weight_analysis",
    #     vectors=True,
    #     mp_fit=True,
    # )

    # Log model analysis results
    # try:
    #     mlflow.log_metrics({
    #         "stable_rank": watcher_results["stable_rank"].mean(),
    #         "condition_number": watcher_results["condition_number"].mean(),
    #         "norm": watcher_results["norm"].mean(),
    #     })
    #     mlflow.log_dict(watcher_results.to_dict(), "weight_analysis.json")
    # except Exception as e:
    #     logging.warning(f"Warning: Weight analysis logging failed: {str(e)}")

    # Save model and artifacts
    logging.info("Saving model and artifacts...")
    try:
        mlflow.pytorch.log_model(
            unwrapped_model,
            "model",
            registered_model_name="review_rating_model",
            pip_requirements=[
                "torch",
                "transformers",
                f"scikit-learn=={sklearn.__version__}",
                "weightwatcher",  # Note: corrected from weightwatch
            ],
            code_paths=["src/review_rating/pipelines/data_science/nodes.py"],
            input_example=next(iter(train_dataset)),
        )
        mlflow.log_dict(unwrapped_model.config.to_dict(), "model_config.json")
    except Exception as e:
        logging.warning(f"Warning: Model saving failed: {str(e)}")

    return unwrapped_model


def evaluate_model(model, test_dataset):
    """
    Evaluate the trained model on test dataset.
    Args:
        model: Trained model
        test_dataset: PyTorch dataset for testing/validation
    Returns:
        classification report
    """
    logging.info("Starting model evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If model is a state dict, load it into a new model
    if isinstance(model, dict) or isinstance(model, OrderedDict):
        eval_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=5,
            problem_type="single_label_classification"
        )
        eval_model.load_state_dict(model)
        model = eval_model

    model = model.to(device)
    predictions, true_labels = [], []

    # Evaluate in batches
    logging.info(f"Evaluating on {len(test_dataset)} samples...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            outputs = model(
                input_ids=item["input_ids"].unsqueeze(0).to(device),
                attention_mask=item["attention_mask"].unsqueeze(0).to(device),
            )
            pred = outputs.logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(pred + 1)
            true_labels.append(item["labels"].item() + 1)

    # Generate and log detailed evaluation metrics
    logging.info("Generating evaluation report...")
    report = classification_report(true_labels, predictions)
    mlflow.log_text(report, "classification_report.txt")

    # Log detailed metrics
    metrics = classification_report(true_labels, predictions, output_dict=True)
    try:
        mlflow.log_metrics({
            f"{k}_f1-score": v["f1-score"]
            for k, v in metrics.items()
            if isinstance(v, dict)
        })
        logging.info("Evaluation metrics logged successfully")
    except Exception as e:
        logging.warning(f"Failed to log evaluation metrics: {str(e)}")

    return report
