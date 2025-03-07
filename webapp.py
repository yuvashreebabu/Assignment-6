import streamlit as st
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

# Streamlit UI
st.title("Titanic Survival Prediction")

Pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare Paid", min_value=0.0, value=50.0)
Embarked = st.selectbox("Port of Embarkation (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)", [0, 1, 2])

if st.button("Predict"):
    result = predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    st.write(f"Prediction: {result}")
