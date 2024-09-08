import numpy as np
import pickle

class MLP:
    activations = []
    def __init__(self, layers=[24, 24], learning_rate=0.01, epochs=100, batch_size=32):
        self.layers = layers
        self.output_layer_index = len(self.layers) - 2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def feedforward(self, X):
        self.activations = [X]
        pre_activation_list = []
        for i in range(len(self.weights)):
            pre_activation = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            pre_activation_list.append(pre_activation)
            if i == self.output_layer_index:
                activations = self.softmax(pre_activation)
            else:
                activations = self.sigmoid(pre_activation)
            self.activations.append(activations)
        return self.activations[-1]

    def compute_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def backpropagation(self, X, y):
        m = y.shape[0]
        dz = self.activations[-1] - y
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train_keep(self, X_train, y_train):
        self.losses_train = []
        self.accuracies_train = []
        self.losses_valid = []
        self.accuracies_valid = []
        
        for epoch in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                y_batch_pred = self.feedforward(X_batch)
                self.backpropagation(X_batch, y_batch)

            y_train_pred = self.feedforward(X_train)
            train_loss = self.compute_loss(y_train, y_train_pred)
            train_accuracy = np.mean(np.argmax(y_train_pred, axis=1) == np.argmax(y_train, axis=1))
            self.losses_train.append(train_loss)
            self.accuracies_train.append(train_accuracy)

            print(f"epoch {epoch+1:02d}/{self.epochs} - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f}")

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        self.losses_train = []
        self.accuracies_train = []
        self.losses_valid = []
        self.accuracies_valid = []
    
        for epoch in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]
                self.feedforward(X_batch)
                self.backpropagation(X_batch, y_batch)

            y_train_pred = self.feedforward(X_train)
            train_loss = self.compute_loss(y_train, y_train_pred)
            train_accuracy = np.mean(np.argmax(y_train_pred, axis=1) == np.argmax(y_train, axis=1))

            self.losses_train.append(train_loss)
            self.accuracies_train.append(train_accuracy)

            if X_valid is not None and y_valid is not None:
                y_valid_pred = self.feedforward(X_valid)
                val_loss = self.compute_loss(y_valid, y_valid_pred)
                val_accuracy = np.mean(np.argmax(y_valid_pred, axis=1) == np.argmax(y_valid, axis=1))
                self.losses_valid.append(val_loss)
                self.accuracies_valid.append(val_accuracy)
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")
    
    def predict(self, X):
        self.feedforward(X)
        y_pred_prob = self.activations[-1]  
        return y_pred_prob

    def predict_keep(self, X):
        y_pred_prob = self.feedforward(X) 
        return np.argmax(y_pred_prob, axis=1) 
    

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'layers': self.layers,
                'weights': self.weights,
                'biases': self.biases,
                'losses_train': self.losses_train,
                'accuracies_train': self.accuracies_train,
                'losses_valid': self.losses_valid,
                'accuracies_valid': self.accuracies_valid
            }, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            
            model = pickle.load(f)
            self.layers = model['layers']
            self.weights = model['weights']
            self.biases = model['biases']
            self.losses_train = model['losses_train']
            self.accuracies_train = model['accuracies_train']
            self.losses_valid = model['losses_valid']
            self.accuracies_valid = model['accuracies_valid']

    def plot_learning_curves(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.losses_train, label='training loss')
        plt.plot(self.losses_valid, label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies_train, label='training acc')
        plt.plot(self.accuracies_valid, label='validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()
