import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.Perceptron import MLP
from src.utils import load_scaler, preprocess_features, preprocess_target, preprocess_data

def main():
    df = pd.read_csv("data_test.csv", header=None)
    print("> dataset loaded from 'data_test.csv'")
    
    X, y_true = preprocess_data(df)
    print("> data preprocessed")

    scaler = load_scaler("scaler.pkl")
    X_scaled = scaler.transform(X)
    print("> data scaled")
    
    mlp = MLP()
    mlp.load_model("./mlp_model.pkl")

    predictions_probabilities = mlp.predict(X_scaled)
    
    y_pred = np.argmax(predictions_probabilities, axis=1)
    
    val_loss = mlp.compute_loss(y_true, predictions_probabilities)
    print(f"Validation loss: {val_loss:.4f}")
    mlp.get_all_metrics(np.argmax(y_true, axis=1), y_pred)
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
        exit(1)
