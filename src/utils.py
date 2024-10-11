import argparse
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a multilayer perceptron.')
    parser.add_argument('--layers', type=int, nargs='+', required=False, default=[24, 24], help='Number of neurons in each layer, e.g., 24 24 24.')
    parser.add_argument('--epochs', type=int, required=False, default=8000, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001, help='Learning rate for training.')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='Batch size for training.')
    parser.add_argument('--loss', type=str, required=False, default='binaryCrossEntropy', help='Loss function to use (binaryCrossentropy).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    return args

def balance_data(X, y):
    smote = SMOTE(sampling_strategy='minority')
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

def preprocess_target(df):
    label_encoder = LabelEncoder()
    y = df.iloc[:, 0].values
    y = label_encoder.fit_transform(y)
    y = np.eye(2)[y]

    return y

def preprocess_features(df):
    df = df.drop(df.columns[0], axis=1)
    X = df.iloc[:, 1:].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def preprocess_data(df):
    df = df.dropna()
    df = df.drop(df.columns[0], axis=1)
    features = preprocess_features(df)
    target = preprocess_target(df)
    
    smote = SMOTE(random_state=42)
    features_resampled, target_resampled = smote.fit_resample(features, target)

    target_resampled_flat = target_resampled.flatten()
    num_classes = len(np.unique(target))
    target_resampled_one_hot = np.eye(num_classes)[target_resampled_flat]

    print(target_resampled_one_hot)
    print(target_resampled_one_hot.shape)

    return features, target