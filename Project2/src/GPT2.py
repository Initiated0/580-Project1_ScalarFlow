import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
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

root = '/work/nayeem/580_project2'

df = pd.read_csv(f'{root}/IMDB Dataset.csv')
print('IMDB Dataset 50k Movie Reviews')
sys.stdout.flush()

# 1. Data Preparation
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset
label_mapping = {'positive': 1, 'negative': 0}
df['sentiment'] = df['sentiment'].map(label_mapping)

# Tokenization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token; use EOS token as pad

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

# 2. Model Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
sys.stdout.flush()
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id

model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 20
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0, 0
    start_time = time.time()
    
    # Training Loop with tqdm
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs.logits, dim=1)
        train_correct += (preds == labels).sum().item()

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_correct / len(train_dataset))

    # Validation Loop with tqdm
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            val_correct += (preds == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_correct / len(val_dataset))

    end_time = time.time()
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, "
          f"Validation Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
          f"Val Acc: {val_accuracies[-1]:.4f}, Time: {end_time - start_time:.2f}s")
    sys.stdout.flush()

# Save Metrics to CSV
metrics_df = pd.DataFrame({
    'epoch': list(range(1, num_epochs + 1)),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_accuracy': train_accuracies,
    'val_accuracy': val_accuracies
})
metrics_df.to_csv(f'{root}/training_metrics_gpt220.csv', index=False)
print("Metrics saved to 'training_metrics_gpt220.csv'")
sys.stdout.flush()

# Testing the Model on the Test Set
model.eval()
true_labels, predictions = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        true_labels.extend(labels.cpu().numpy())
        predictions.extend(preds.cpu().numpy())

cm = confusion_matrix(true_labels, predictions)

# Create Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save Confusion Matrix to a File
confusion_matrix_path = f'{root}/confusion_matrix_gpt220.png'
plt.savefig(confusion_matrix_path)
print(f"Confusion matrix saved to '{confusion_matrix_path}'")
sys.stdout.flush()
# Close the plot to free up memory if running in a script
plt.close()

# Save the Fine-Tuned Model
model_save_path = f'{root}/gpt2_finetuned20.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Fine-tuned model saved to '{model_save_path}'")
print('***Antahaa***')
sys.stdout.flush()
