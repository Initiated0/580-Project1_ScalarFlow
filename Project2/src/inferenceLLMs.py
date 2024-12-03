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


# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


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


# Load the fine-tuned model
modelTuned = 'distilbert_finetuned10'
model_path = f'{root}/{modelTuned}.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

model = model.to(device)
model.eval()
print("Model successfully loaded and ready for inference!")
foo()

# Perform inference on the test set
predictions, true_labels, test_texts = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Running Inference on Test Data"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)  # Ground truth labels
        texts = batch['text']  # Extract original texts

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().numpy())  # Convert to numpy
        true_labels.extend(labels.cpu().numpy())  # Convert to numpy
        test_texts.extend(texts)  # Collect original texts

# Save the results along with original test texts
results_df = pd.DataFrame({
    'review': test_texts,
    'true_label': true_labels,
    'predicted_label': predictions
})

# Save to a CSV file
results_csv_path = f'{root}/test_results_{modelTuned}.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"Test results saved to '{results_csv_path}'")


# Compute Confusion Matrix
cm = confusion_matrix(true_labels, predictions)

# Generate Classification Report for the Current Model
report = classification_report(true_labels, predictions, target_names=['Negative', 'Positive'], output_dict=True)

# Convert the report to a DataFrame for better visualization and saving
report_df = pd.DataFrame(report).transpose()

# Save the report to a CSV file
precision_recall_f1_path = f'{root}/precision_recall_f1_scores_{modelTuned}.csv'
report_df.to_csv(precision_recall_f1_path)
print(f"Precision, Recall, and F1-scores saved to '{precision_recall_f1_path}'")

# Display the report
print("Precision, Recall, and F1-Scores:")
print(report_df)
# Create a Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix of {modelTuned}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save Confusion Matrix to a File
confusion_matrix_path = f'{root}/confusion_matrix_{modelTuned}.png'
plt.savefig(confusion_matrix_path)
print(f"Confusion matrix saved to '{confusion_matrix_path}'")
plt.close()  # Close the plot to free up memory if running in a script
foo()

