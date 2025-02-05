# **Credict Card Fraud Detection Project**

# Data From Kaggle
# https://www.kaggle.com/code/renjithmadhavan/credit-card-fraud-detection-using-python/notebook#How-many-are-fraud-and-how-many-are-not-fraud-?

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
file_path = "filepath" # Change to filepath; Temporarily Disabled
data = pd.read_csv(file_path)

# First 5 rows
print(data.head())

# Statistics
data.describe()

# Identify Fraud/Not Fraud Transactions
class_names = {0: 'Not Fraud', 1: 'Fraud'}
print(data.Class.value_counts().rename(index=class_names))

# Scaling 'Amount' and 'Time'
data['scaled_amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1, 1))

# Data Splitting
X = data.drop(['Class'], axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate Accuracy and F1-score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f}, F1-score: {f1:.4f})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.text(0.5, -0.2, f'Accuracy: {accuracy:.4f}', ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.5, -0.3, f'F1-score: {f1:.4f}', ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)

plt.show()

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"F1-score: {f1}")

# Darian009