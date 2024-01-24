import numpy as np

# class LinearRegression:
#     def __init__(self):
#         self.coefficients = None

#     def fit(self, X, y, epochs = None):
#         # Add a column of ones to the input features for the intercept term
#         X_extended = np.c_[np.ones(X.shape[0]), X]
#         # Use the pseudo-inverse to calculate the coefficients
#         self.coefficients = np.linalg.pinv(X_extended.T @ X_extended) @ X_extended.T @ y
#         return self

#     def predict(self, X):
#         if self.coefficients is None:
#             raise ValueError("Model has not been trained. Call fit() first.")

#         # Add a column of ones to the input features for the intercept term
#         X_extended = np.c_[np.ones(X.shape[0]), X]
#         predictions = X_extended @ self.coefficients

#         return predictions


####################################################################################3

# import numpy as np

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coefficients = None

    def _compute_cost(self, X, y):
        m = len(y)
        predictions = X @ self.coefficients
        error = predictions - y
        cost = (1 / (2 * m)) * np.sum(error**2)
        return cost

    def fit(self, X, y, epochs=None):
        y = y.squeeze()
        
        # Feature scaling - normalize each feature
        X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Add a column of ones to the input features for the intercept term
        X_extended = np.c_[np.ones(X_normalized.shape[0]), X_normalized]
        m, n = X_extended.shape

        # Initialize coefficients randomly
        self.coefficients = np.random.rand(n)

        for _ in range(self.epochs):
            predictions = X_extended @ self.coefficients
            error = predictions - y
            gradient = (1 / m) * X_extended.T @ error
            self.coefficients -= self.learning_rate * gradient

        return self

    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Feature scaling for prediction
        X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

        # Add a column of ones to the input features for the intercept term
        X_extended = np.c_[np.ones(X_normalized.shape[0]), X_normalized]
        predictions = X_extended @ self.coefficients

        return predictions


############################################################################