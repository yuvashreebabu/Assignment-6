import pickle
import numpy as np

# Load model and scaler
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Function for prediction
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return "Survived" if prediction[0] == 1 else "Did not Survive"

# Example Prediction
result = predict_survival(3, 1, 22, 1, 0, 7.25, 2)
print("Prediction:", result)
