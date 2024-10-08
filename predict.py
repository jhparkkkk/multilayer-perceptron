import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.Perceptron import MLP

def main():
    df_valid = pd.read_csv("data/valid_data.csv")
    df_valid.replace({'B': 0, 'M': 1}, inplace=True)
    
    y_valid = df_valid.iloc[:, 0].values 
    y_valid_one_hot = np.eye(2)[y_valid]  
    X_valid = df_valid.iloc[:, 1:].values

    mlp = MLP()
    mlp.load_model("data/mlp_model.pkl")

    y_valid_pred_prob = mlp.predict(X_valid)
    y_valid_pred = np.argmax(y_valid_pred_prob, axis=1)

    val_loss = mlp.compute_loss(y_valid_one_hot, y_valid_pred_prob)
    val_accuracy = np.mean(y_valid_pred == y_valid)

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
        exit(1)
