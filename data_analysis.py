import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    data = pd.read_csv("data/data.csv", header=None)
    print(data.head())
    #print(data.info())
    #print(data.describe())

    # delete first column
    data = data.drop(data.columns[0], axis=1)

    breast_cancer = load_breast_cancer()
    column_headers = list(breast_cancer.feature_names)
    column_headers = [np.str_("diagnosis")] + column_headers
    data.columns = column_headers

    print(data.head())
    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

    plt.figure(figsize=(6, 6))
    dmap = {0: "Begnin", 1: "Malignant"}
    diagnosis_counts = data["diagnosis"].value_counts().rename(dmap)
    diagnosis_counts.plot(kind="bar", color=sns.color_palette("pastel"))

    total = diagnosis_counts.sum()
    percentages = diagnosis_counts / total * 100

    for i, (count, percentage) in enumerate(
        zip(diagnosis_counts.values, percentages.values)
    ):
        plt.text(i, count + 5, f"{percentage:.2f}%", ha="center", fontsize=12)

    plt.title("Malignant vs Begnin Data Points")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of data points")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # heatmap
    #data = data.drop("diagnosis", axis=1)

    x = data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    data = pd.DataFrame(x_scaled, columns=x.columns)

    print(data.head())
    plt.figure(figsize=(20, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, linewidths=.5, fmt= '.1f', cmap="rocket")
    plt.title("Heatmap showing correlation between features")
    plt.tight_layout()
    plt.show()
