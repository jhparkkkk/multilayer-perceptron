import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np
import seaborn as sns
import warnings

# ignore all warnings
warnings.filterwarnings("ignore")

from src.analyze.plot import plot_violin
from src.analyze.plot import plot_swarm
from src.analyze.plot import plot_heatmap
from src.analyze.plot import plot_count


if __name__ == "__main__":
    data = pd.read_csv("data/data.csv", header=None)
    print(data.describe())

    # delete first column
    data = data.drop(data.columns[0], axis=1)

    # add headers
    breast_cancer = load_breast_cancer()
    column_headers = list(breast_cancer.feature_names)
    column_headers = [np.str_("diagnosis")] + column_headers
    data.columns = column_headers

    # encode diagnosis values
    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})
    plot_count(data=data)

    # extract x and y
    y = data.diagnosis
    x = data.drop("diagnosis", axis=1)

    # standardize x
    x_scaled = (x - x.mean()) / (x.std())
    data = pd.DataFrame(x_scaled, columns=x.columns)

    # extract mean, square error and worst
    df_mean = data[data.columns[:10]]
    df_se = data.drop(data.columns[:10], axis=1)
    df_se = df_se.drop(df_se.columns[10:], axis=1)
    df_worst = data.drop(data.columns[:20], axis=1)

    # Plot violin
    plot_violin(df_mean, y, "mean")
    plot_violin(df_se, y, "error")
    plot_violin(df_worst, y, "worst")

    # Plot swarm
    plot_swarm(df_mean, y, "mean")
    plot_swarm(df_se, y, "error")
    plot_swarm(df_worst, y, "worst")

    # Plot heatmap
    plot_heatmap(data)
    plt.tight_layout()
    plt.show()
