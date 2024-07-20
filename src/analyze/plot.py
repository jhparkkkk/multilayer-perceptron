import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def plot_count(data):
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
    plt.savefig('diagnosis_counter.png')


def plot_violin(x, y, feature_category: str):
    data = pd.concat([y,x],axis=1)
    data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
    plt.figure(figsize=(10,10))
    sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.title(f"Violin plot of {feature_category} features")
    plt.savefig(f'violin_plot_{feature_category}.png')


def plot_swarm(x, y, feature_category: str):
    sns.set_theme(style="whitegrid", palette="muted")
    data = pd.concat([y,x],axis=1)
    data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
    plt.figure(figsize=(10,10))
    sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
    plt.xticks(rotation=90)
    plt.title(f"Swarm plot of {feature_category} features")
    plt.savefig(f'swarm_plot_{feature_category}.png')

def plot_heatmap(features):
    plt.figure(figsize=(20, 10))
    correlation_matrix = features.corr()
    sns.heatmap(correlation_matrix, annot=True, linewidths=.5, fmt= '.1f', cmap="rocket")
    plt.title("Heatmap showing correlation between features")
    plt.savefig(f'heatmap.png')
