# import os
# import wandb
import numpy as np
from sklearn.metrics import accuracy_score
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import MinMaxScaler

class MLR:
    def __init__(self, lr=0.01, epochs=2000, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size       
    def init_params(self, X, y):
        self.weights = np.random.randn(y.shape[1], X.shape[1]) # (no of classes, no of features)
        self.k = y.shape[1]
        self.n = X.shape[0]
        self.m = X.shape[1]
        
    def fit(self, X, y, epochs=None, X_val=None, y_val=None):
        X = np.insert(X, 0, 1, axis=1) # insert 1 for bias
        
        self.init_params(X, y)
        
        self.loss = []
        self.accuracy = []
        for i in range(self.epochs):
            # Shuffle the data
            indices = np.random.permutation(self.n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Split the data into batches
            for j in range(0, self.n, self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                y_batch = y_shuffled[j:j+self.batch_size]
                
                # Calculate the gradient and update the weights
                soft = self.softmax(self.calculate_z(X_batch))
                grad = self.gradient(X_batch, y_batch, soft)
                self.weights -= self.lr * grad
            
            # Calculate the loss and store it
            soft = self.softmax(self.calculate_z(X))
            self.loss.append(self.cross_entropy_loss(y, soft))
            # print(f"Epoch {i+1}: loss = {self.loss[-1]}")
            # print(X_val.shape)
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                # print(y_pred[:5])
                
                # print(y_pred.shape)
                self.accuracy.append(accuracy_score(y_true = np.argmax(y_val , axis=1), y_pred= y_pred))
                # print(self.accuracy[-1])
                # print(classification_report(y_true = np.argmax(y_val , axis=1), y_pred= y_pred, zero_division=1))
                
        # return self.weights, self.loss
    
    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True) # for numerical stability
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    def calculate_z(self, X):
        return np.dot(X, self.weights.T)
    
    def cross_entropy_loss(self, y, y_hat):
        loss = -np.sum(y * np.log(y_hat)) / y.shape[0]
        return loss
    
    def gradient(self, X, y, y_hat):
        grad = np.dot((y_hat - y).T, X) / X.shape[0]
        return grad

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        z = self.calculate_z(X)
        y_pred = self.softmax(z)
        # y_pred = np.argmax(self.softmax(z), axis=1)
        # print(y_pred.shape)
        return y_pred

# MLR = MultiomialLogisticRegression(epochs=2000)
# MLR.fit(X_train, y_train, X_val, y_val)
# # MLR.fit(X_train, y_train)
