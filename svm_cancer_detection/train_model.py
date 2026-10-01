import pickle
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("1. Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0 = malignant, 1 = benign

print("2. Splitting into Train (80%) and Test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Performing Feature Selection (ANOVA F-test) on Training Data ONLY...")
# Select top 5 features
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X_train, y_train)

# Get selected feature names
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask].tolist()

print("\n--- Selected Features ---")
for i, f in enumerate(selected_features, 1):
    print(f"{i}. {f}")
print("-------------------------\n")

# Transform datasets to keep only the 5 selected features
X_train_selected = X_train.loc[:, selected_features]
X_test_selected = X_test.loc[:, selected_features]

print("4. Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

print("5. Training SVM Model...")
svm_model = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

print("6. Evaluating Model on Untouched Test Set...")
y_pred = svm_model.predict(X_test_scaled)
print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Malignant (0)", "Benign (1)"]))
print("------------------------\n")

print("7. Extracting Sample Data for Frontend...")
# Get one Malignant (0) and one Benign (1) sample from the training set
malignant_sample = X_train_selected[y_train == 0].iloc[0].to_dict()
benign_sample = X_train_selected[y_train == 1].iloc[0].to_dict()

sample_data = {
    "malignant": malignant_sample,
    "benign": benign_sample
}

print("8. Saving Models and Artifacts...")
with open("model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)
with open("sample_data.json", "w") as f:
    json.dump(sample_data, f)
    
print("Success! Saved model.pkl, scaler.pkl, selected_features.pkl, and sample_data.json.")
