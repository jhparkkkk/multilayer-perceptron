import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a multilayer perceptron.')
    parser.add_argument('--layers', type=int, nargs='+', required=False, default=[24, 24], help='Number of neurons in each layer, e.g., 24 24 24.')
    parser.add_argument('--epochs', type=int, required=False, default=1000, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.1, help='Learning rate for training.')
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='Batch size for training.')
    parser.add_argument('--loss', type=str, required=False, default='binaryCrossEntropy', help='Loss function to use (binaryCrossentropy).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    return args
