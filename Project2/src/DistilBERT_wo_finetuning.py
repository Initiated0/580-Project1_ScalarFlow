import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import sys
import pickle

print(f'Torch available: {torch.cuda.is_available()}')
print(f'# of Devices: {torch.cuda.device_count()}')

def foo():
    print('\n\n\n')

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

print(f"Running on {device.upper()}")



root = '/work/nayeem/580_project2'

df = pd.read_csv(f'{root}/IMDB Dataset.csv')
print('IMDB Dataset 50k Movie Reviews')
sys.stdout.flush()
# 1. Data Preparation
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset
label_mapping = {'positive': 1, 'negative': 0}
df['sentiment'] = df['sentiment'].map(label_mapping)


# Load the pretrained DistilBERT model (without fine-tuning)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model = model.to(device)
model.eval()
print("Plain DistilBERT model loaded successfully.")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'text': text,  # Include the original text
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }



# 1. Data Preparation: Train/Validation/Test Split
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, stratify=df['sentiment'], random_state=42
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# Convert to lists for tokenization
train_texts, val_texts, test_texts = train_texts.tolist(), val_texts.tolist(), test_texts.tolist()
train_labels, val_labels, test_labels = train_labels.tolist(), val_labels.tolist(), test_labels.tolist()

# Tokenization and Dataset Preparation remain the same
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)


# Perform inference on the test set
predictions, true_labels, test_texts = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Running Inference with Plain DistilBERT"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        texts = batch['text']

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        test_texts.extend(texts)

# Save the results along with original test texts
results_df = pd.DataFrame({
    'review': test_texts,
    'true_label': true_labels,
    'predicted_label': predictions
})
results_csv_path = f'{root}/plain_distilbert_test_results.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"Test results saved to: {results_csv_path}")

# Generate Confusion Matrix and Classification Report
cm = confusion_matrix(true_labels, predictions)

# Classification Report
report = classification_report(true_labels, predictions, target_names=['Negative', 'Positive'])
print("\nClassification Report:")
print(report)

# Save the Classification Report
report_dict = classification_report(true_labels, predictions, target_names=['Negative', 'Positive'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_csv_path = f'{root}/plain_distilbert_classification_report.csv'
report_df.to_csv(report_csv_path, index=True)
print(f"Classification report saved to: {report_csv_path}")

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Plain DistilBERT - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the Confusion Matrix Plot
confusion_matrix_path = f'{root}/plain_distilbert_confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.close()
print(f"Confusion matrix saved to: {confusion_matrix_path}")
