import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path, header=None)
    # data = data.reset_index(drop=True)
    # data = data.iloc[: , 1:]
    # data = data.iloc[1:]
    print(data)
    # data.replace({'B': 0, 'M': 1}, inplace=True)
    return data


def split_and_save_data(data, seed=42, test_size=0.2, save_dir="data"):
    y = data.iloc[:, [1] ]
    # y = data.drop(data.columns[1], axis=1, inplace=True)
    data = data.drop(data.columns[[1]], axis=1) 
    X = data
    print(X)
    print('-------------------------------')
    print(y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    print(X_train)
    print(X_valid)

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_valid_scaled = scaler.transform(X_valid)

    # smote = SMOTE(random_state=seed)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=X.columns)
    # X_train_scaled = pd.DataFrame(X_train_resampled, columns=X.columns)

    # y_train_resampled = pd.Series(y_train_resampled, name="diagnosis")
    # y_valid.reset_index(drop=True, inplace=True)

    # # X_valid_scaled.insert(0, "diagnosis", y_valid)
    # valid_data = X_valid_scaled

    # # X_train_scaled.insert(0, "diagnosis", y_train_resampled)
    # train_data = X_train_scaled

    train_data = pd.concat([X_train, y_train], axis=1)
    valid_data = pd.concat([X_valid, y_valid], axis=1)

    train_data = X_train.insert(1, '', y_train[0])
    print('train_data', train_data)
    col = data.pop()
    data.insert(1, col.name, col)
    train_data.to_csv(f"data_training.csv", index=False)
    valid_data.to_csv(f"data_test.csv", index=False)

    print("Data has been split and saved successfully.")
    print(f"Number of training samples : {train_data.shape[0]}")
    print(f"Number of validation samples : {valid_data.shape[0]}")


if __name__ == "__main__":
    # try:
        file_path = "data.csv"

        data = load_and_prepare_data(file_path)

        split_and_save_data(data)
    # except Exception as error:
        # print(error)
        # exit(1)
