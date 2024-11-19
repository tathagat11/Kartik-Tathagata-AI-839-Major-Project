import torch
import mlflow
from transformers import RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids + 1
    preds = pred.predictions.argmax(-1) + 1
    return {'classification_report': classification_report(labels, preds)}

def train_model(train_dataset, test_dataset, params):
    mlflow.log_param("train_size", len(train_dataset))
    mlflow.log_param("test_size", len(test_dataset))

    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=5,
        problem_type="single_label_classification"
    )

    raw_scores = [item['labels'].cpu().item() for item in train_dataset]
    class_counts = np.bincount(raw_scores)
    mlflow.log_param("class_distribution", dict(enumerate(class_counts.tolist())))
    
    training_args = TrainingArguments(**params)
    trainer = Trainer(model=model, args=training_args,
                     train_dataset=train_dataset,
                     eval_dataset=test_dataset,
                     compute_metrics=compute_metrics)
    
    trainer.train()
    mlflow.pytorch.log_model(
        model, 
        "model",
        registered_model_name="review_rating_model",
    )
    return model

def evaluate_model(model, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            outputs = model(input_ids=item['input_ids'].unsqueeze(0).to(device),
                          attention_mask=item['attention_mask'].unsqueeze(0).to(device))
            pred = outputs.logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(pred + 1)
            true_labels.append(item['labels'].item() + 1)
    
    report = classification_report(true_labels, predictions)
    mlflow.log_text(report, "classification_report.txt")
    metrics = classification_report(true_labels, predictions, output_dict=True)
    mlflow.log_metrics({f"{k}_f1-score": v['f1-score'] 
                       for k, v in metrics.items() if isinstance(v, dict)})
    return report