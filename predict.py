import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.Perceptron import MLP
from src.utils import preprocess_features, preprocess_target

def main():
    df = pd.read_csv("data_test.csv")
    df = df.dropna()

    df = df.drop(df.columns[0], axis=1)

    y_valid = preprocess_target(df)

    print(df)
    X_valid = preprocess_features(df)
    X_valid =  df.drop(df.columns[0], axis=1)
    
    print(type(X_valid))
    X_valid = X_valid.values

    print(y_valid)
    print(X_valid)
    print(type(X_valid))

    
    mlp = MLP()
    mlp.load_model("./mlp_model.pkl")

    print(X_valid)
    print(X_valid.shape)
    y_valid_pred_prob = mlp.predict(X_valid)
    y_valid_pred = np.argmax(y_valid_pred_prob, axis=1)

    val_loss = mlp.compute_loss(y_valid, y_valid_pred_prob)
    y_valid = np.argmax(y_valid, axis=1)
    val_accuracy = np.mean(y_valid_pred == y_valid)

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    # try:
        main()
    # except Exception as error:
        # print(error)
        # exit(1)
