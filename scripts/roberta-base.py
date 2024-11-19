import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy.special import softmax
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Initialize tokenizer and model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def get_sentiment_scores(text):
    # Encode text
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    # Get model output
    with torch.no_grad():
        output = model(**encoded)
    
    # Get sentiment scores
    scores = output[0][0].cpu().numpy()
    scores = softmax(scores)
    
    return scores

# Load data
print("Loading data...")
df = pd.read_csv("data/01_raw/Reviews.csv")
df = df.head(5000)

# Process reviews to get sentiment features
print("Processing reviews...")
sentiment_features = []
scores = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        sentiment_scores = get_sentiment_scores(row['Text'])
        sentiment_features.append(sentiment_scores)
        scores.append(row['Score'])
    except Exception as e:
        print(f"Error processing review {row['Id']}: {e}")
        sentiment_features.append(None)
        scores.append(None)

# Convert to numpy arrays and handle any failed processing
sentiment_features = np.array([f for f in sentiment_features if f is not None])
scores = np.array([s for s in scores if s is not None])

# Split data for classifier training
X_train, X_test, y_train, y_test = train_test_split(
    sentiment_features, scores, test_size=0.2, random_state=42
)

# Train classifier
print("Training classifier...")
classifier = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    class_weight='balanced'
)
classifier.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
predictions = classifier.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
))

# Function for single prediction
def predict_single_review(text, classifier):
    sentiment_scores = get_sentiment_scores(text)
    rating = classifier.predict([sentiment_scores])[0]
    proba = classifier.predict_proba([sentiment_scores])[0]
    return rating, proba

# Example usage
example_text = "This product is amazing! Highly recommended!"
predicted_rating, probabilities = predict_single_review(example_text, classifier)
print(f"\nExample prediction:")
print(f"Text: {example_text}")
print(f"Predicted rating: {predicted_rating} stars")
print("Rating probabilities:")
for rating, prob in enumerate(probabilities, 1):
    print(f"{rating} stars: {prob:.3f}")

# Save the classifier
joblib.dump(classifier, 'sentiment_classifier.joblib')

# Analysis of classifier performance
print("\nDetailed Analysis:")

# Feature importance
print("\nFeature Importance:")
feature_names = ['Negative', 'Neutral', 'Positive']
for i, feature in enumerate(feature_names):
    importance = np.abs(classifier.coef_[:, i]).mean()
    print(f"{feature}: {importance:.3f}")

# Confusion analysis
print("\nMost confident correct predictions:")
for true_rating in range(1, 6):
    mask = (y_test == true_rating)
    if not any(mask):
        continue
    probs = classifier.predict_proba(X_test[mask])
    confidences = np.max(probs, axis=1)
    most_confident_idx = np.argmax(confidences)
    most_confident_prob = confidences[most_confident_idx]
    print(f"\nRating {true_rating}:")
    print(f"Confidence: {most_confident_prob:.3f}")
    print(f"Probabilities: {probs[most_confident_idx]}")

# Save full results
results_df = df.copy()
all_predictions = classifier.predict(sentiment_features)
results_df['predicted_rating'] = all_predictions
results_df['prediction_correct'] = results_df['Score'] == results_df['predicted_rating']

# Add prediction probabilities
probabilities = classifier.predict_proba(sentiment_features)
for i in range(5):
    results_df[f'probability_{i+1}_star'] = probabilities[:, i]

print("\nOverall accuracy:", (results_df['prediction_correct'].mean() * 100))

# Save results
results_df.to_csv('data/08_reporting/prediction_results.csv', index=False)