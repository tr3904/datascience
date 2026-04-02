#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from preprocess import load_data, preprocess_data

# Load data
df = load_data("churn_data.csv")

# Preprocess
X, y = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("../model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

