import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.Perceptron import MLP

if __name__ == "__main__":
    df_valid = pd.read_csv("data/valid_data.csv")
    y_valid = df_valid.iloc[:, 0].values  # True labels
    y_valid_one_hot = np.eye(2)[y_valid]  # One-hot encode the labels for loss computation
    X_valid = df_valid.iloc[:, 1:].values  # Features

    mlp = MLP()
    mlp.load_model("data/mlp_model.pkl")

    y_valid_pred_prob = mlp.predict(X_valid)  # Get predicted probabilities (from softmax)
    y_valid_pred = np.argmax(y_valid_pred_prob, axis=1)

    val_loss = mlp.compute_loss(y_valid_one_hot, y_valid_pred_prob)
    val_accuracy = np.mean(y_valid_pred == y_valid)

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    mlp.plot_learning_curves()
