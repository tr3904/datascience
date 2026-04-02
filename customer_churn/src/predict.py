import pickle
import pandas as pd

# Load model
with open("../model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_new(data_dict):
    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)
    return "Churn" if prediction[0] == 1 else "No Churn"


sample = {
    "gender": 1,
    "tenure": 12,
    "MonthlyCharges": 70,
    "TotalCharges": 840,
    "Contract": 0
}

print(predict_new(sample))
