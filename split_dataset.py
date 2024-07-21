import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path, header=None)

    # load headers names
    breast_cancer = load_breast_cancer()
    column_headers = ["id", "diagnosis"] + list(breast_cancer.feature_names)
    data.columns = column_headers

    # delete `id` column
    data = data.drop(columns=["id"])

    # Map diagnostic values
    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

    return data


def split_and_save_data(data, seed=42, test_size=0.2, save_dir="data"):
    # Split x and y
    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]

    # Split into train and test dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Apply SMOTE to balance the training set
    smote = SMOTE(random_state=seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=X.columns)
    X_train_scaled = pd.DataFrame(X_train_resampled, columns=X.columns)

    y_train_resampled = pd.Series(y_train_resampled, name="diagnosis")
    y_valid.reset_index(drop=True, inplace=True)

    X_valid_scaled.insert(0, "diagnosis", y_valid)
    valid_data = X_valid_scaled

    X_train_scaled.insert(0, "diagnosis", y_train_resampled)
    train_data = X_train_scaled

    train_data.to_csv(f"{save_dir}/train_data.csv", index=False)
    valid_data.to_csv(f"{save_dir}/valid_data.csv", index=False)

    print("Data has been split and saved successfully.")
    print(f"Number of training samples : {train_data.shape[0]}")
    print(f"Number of validation samples : {valid_data.shape[0]}")


if __name__ == "__main__":
    file_path = "data/data.csv"

    data = load_and_prepare_data(file_path)

    split_and_save_data(data)
