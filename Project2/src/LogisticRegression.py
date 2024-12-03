import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
root = '/work/nayeem/580_project2'
df = pd.read_csv(f'{root}/IMDB Dataset.csv')

# Shuffle the dataset and map sentiment labels to integers
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
label_mapping = {'positive': 1, 'negative': 0}
df['sentiment'] = df['sentiment'].map(label_mapping)

# Split the dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, stratify=df['sentiment'], random_state=42
)

# Bag-of-Words Representation
vectorizer = CountVectorizer(max_features=10000, stop_words='english')  # Limit to top 10,000 features
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, train_labels)

# Predictions
predictions = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(test_labels, predictions)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
report = classification_report(test_labels, predictions, target_names=['Negative', 'Positive'])
print(report)

# Save the Classification Report
report_dict = classification_report(test_labels, predictions, target_names=['Negative', 'Positive'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_csv_path = f'{root}/logistic_regression_classification_report.csv'
report_df.to_csv(report_csv_path, index=True)
print(f"Classification report saved to: {report_csv_path}")

# Confusion Matrix
cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Logistic Regression - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save Confusion Matrix Plot
confusion_matrix_path = f'{root}/logistic_regression_confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.close()
print(f"Confusion matrix saved to: {confusion_matrix_path}")
