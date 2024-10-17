import argparse
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import pickle

from sklearn.preprocessing import LabelEncoder
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a multilayer perceptron.')
    parser.add_argument('--layers', type=int, nargs='+', required=False, default=[24, 24], help='Number of neurons in each layer, e.g., 24 24 24.')
    parser.add_argument('--epochs', type=int, required=False, default=8000, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001, help='Learning rate for training.')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='Batch size for training.')
    parser.add_argument('--loss', type=str, required=False, default='binaryCrossEntropy', help='Loss function to use (binaryCrossentropy).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    return args

def load_scaler(file_path):
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    print("> scaler loaded from 'scaler.pkl'")
    return scaler

def preprocess_target(df):
    y = df.iloc[:, 0].values
    y = np.where(y == 'M', 1, 0)
    y = np.eye(2)[y]
    return y

def preprocess_features(df):
    df = df.drop(df.columns[0], axis=1)
    X = df.values
    return X

def preprocess_data(df):
    df = df.dropna()
    df = df.drop(df.columns[0], axis=1)
    features = preprocess_features(df)
    target = preprocess_target(df)

    return features, target