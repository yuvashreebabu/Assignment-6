import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("Dataset.csv")

# Drop unnecessary columns
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)   # Fill missing ages with median
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # Fill missing Embarked with mode
df["Fare"].fillna(df["Fare"].median(), inplace=True)  # Fill missing Fare with median

# Convert categorical features to numerical
label_enc = LabelEncoder()
df["Sex"] = label_enc.fit_transform(df["Sex"])  # Convert Sex to 0 and 1
df["Embarked"] = label_enc.fit_transform(df["Embarked"])  # Convert Embarked to numerical

# Check if any missing values remain
print("Missing values after preprocessing:\n", df.isnull().sum())

# Select features and target variable
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the preprocessed data & scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save training and test data
np.savez("titanic_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
