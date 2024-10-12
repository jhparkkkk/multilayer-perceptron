import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import parse_arguments
from src.Perceptron import MLP
from src.utils import preprocess_data
def main():
    args = parse_arguments()
    
    np.random.seed(args.seed)

    df = pd.read_csv("data_training.csv")
    
    X, Y = preprocess_data(df)
    # print(type(X))
    # print(type(Y))
    # print(X.shape)
    # print(Y.shape)
    # print(Y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=args.seed)

    layers = [X_train.shape[1]] + args.layers + [y_train.shape[1]]
    mlp = MLP(layers=layers, learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batch_size)
    mlp.train(X_train, y_train, X_valid, y_valid)

    mlp.save_model("mlp_model.pkl")
    print("> saving model './mlp_model.pkl' to disk")

    mlp.plot_learning_curves()


if __name__ == "__main__":
    #try:
        main()
    #except Exception as error:
    #    print(error)
    #    exit(1)
