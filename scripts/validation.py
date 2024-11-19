import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
import numpy as np

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

def evaluate(model, dataset, device):
   model.eval()
   predictions = []
   true_labels = []
   
   with torch.no_grad():
       for i in range(len(dataset)):
           item = dataset[i]
           input_ids = item['input_ids'].unsqueeze(0).to(device)
           attention_mask = item['attention_mask'].unsqueeze(0).to(device)
           
           outputs = model(input_ids=input_ids, attention_mask=attention_mask)
           pred = outputs.logits.argmax(dim=-1).cpu().numpy()
           predictions.extend(pred + 1)
           true_labels.append(item['labels'].item() + 1)
   
   return classification_report(true_labels, predictions)

def main():
   # Load and sample data
   df = pd.read_csv("data/01_raw/Reviews.csv")
   sample_df = df.sample(n=1000, random_state=42)
   
   # Load model and tokenizer
   model_path = "data/08_reporting/results/checkpoint-700"
   model = RobertaForSequenceClassification.from_pretrained(model_path)
   tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
   
   # Setup device
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   
   # Create dataset
   eval_dataset = ReviewDataset(
       sample_df['Text'].values,
       sample_df['Score'].values,
       tokenizer
   )
   
   # Run evaluation
   report = evaluate(model, eval_dataset, device)
   print("\nClassification Report:")
   print(report)

if __name__ == "__main__":
   main()