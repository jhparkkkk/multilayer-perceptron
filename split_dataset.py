import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
        while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
            training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    Raises:
        This function should not raise any Exception.
    """
    if x.__class__ != np.ndarray or y.__class__ != np.ndarray or proportion.__class__ != float:
        return None

    if x.size == 0 or y.size == 0:
        return None

    if x.shape[0] != y.shape[0]:
        return None

    dataset = [list(i) + [j] for i, j in zip(x, y)]
    np.random.shuffle(dataset)

    x_dataset = np.array([i[:-1] for i in dataset])
    y_dataset = np.array([i[-1] for i in dataset])

    split_index = int(x.shape[0] * proportion)

    x_train, x_test = x_dataset[:split_index], x_dataset[split_index:]
    y_train, y_test = y_dataset[:split_index], y_dataset[split_index:]

    return x_train, x_test, y_train, y_test

def split_and_save_data(data, seed=43, test_size=0.2, save_dir="data"):
    data = pd.read_csv(file_path, header=None, skiprows=1)

    y = data.iloc[:, [1] ]
    X = data.drop(data.columns[[1]], axis=1) 

    X_train, X_valid, y_train, y_valid = data_spliter(X.values, y.values, 1 - test_size)


    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=y.columns)
    y_valid = pd.DataFrame(y_valid, columns=y.columns)

    X_train.insert(1, y_train.columns[0], y_train)
    X_valid.insert(1, y_valid.columns[0], y_valid)

    train_data = X_train
    valid_data = X_valid

    train_data.to_csv(f"data_training.csv", index=False, header=False)
    valid_data.to_csv(f"data_test.csv", index=False, header=False)

    print("Data has been split and saved successfully.")
    print(f"Number of training samples : {train_data.shape[0]}")
    print(f"Number of validation samples : {valid_data.shape[0]}")


if __name__ == "__main__":
    try:
        file_path = "data.csv"


        split_and_save_data(file_path)
    except Exception as error:
        print(error)
        exit(1)
