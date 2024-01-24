import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from icecream import ic
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score

DEBUG = False

class Sigmoid: # Sigmoid Function:
    def __init__(self) -> None:
        pass
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def activation(self, z):
        return self.sigmoid(z)
    def activation_derivative(self, s):
        # s = self.sigmoid(z)
        return s * (1 - s)


class Tanh: # Hyperbolic Tangent (Tanh) Function    
    def __init__(self) -> None:
        pass
    def activation(self, z):
        return np.tanh(z)
    
    def activation_derivative(self, t):
        # t = np.tanh(z)
        return 1 - t**2 

        
class ReLU: # Rectified Linear Unit (ReLU)
    def __init__(self) -> None:
        pass
    def activation(self, z):
        return np.maximum(0, z)
    
    def activation_derivative(self, z):
        return (z > 0)

class Identity:
    def __init__(self):
        pass

    def activation(self, z):
        # Linear activation function (identity function)
        return z

    def activation_derivative(self, z):
        # Derivative of the linear activation function is always 1
        # return np.ones_like(z)
        pass
    

# Used for Classification
class Softmax:
    def __init__(self) -> None:
        pass
    def activation(self, z):
        # print(z.shape)
        exps = np.exp(z - np.max(z))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def activation_derivative(self, z):
        pass
    
class CrossEntropyLoss:
    def __init__(self, epsilon = 1e-15) -> None:
        self.epsilon = epsilon
    def loss(self, y_pred, y_true):
        epsilon = 1e-10
        loss = -np.sum(y_true * np.log(y_pred+epsilon)) / y_true.shape[0]
        return loss
    def gradient(self, y_pred, y_true):

        return ((y_pred - y_true) / y_true.shape[0])
    
class MSELoss:
    def __init__(self):
        pass

    def loss(self, y_pred, y_true):
        # Calculate the squared differences between predicted and true values
        squared_errors = (y_pred - y_true) ** 2
        # Compute the mean of squared errors
        mse = np.mean(squared_errors)
        return mse

    def gradient(self, y_pred, y_true):
        # Compute the gradient of the MSE loss
        gradient = 2 * (y_pred - y_true) / len(y_true)
        return gradient
    
class Layer:
    # no_of_neurons is equal to the no_of_output of the layer
    def __init__(self, no_of_inputs, no_of_neurons, activation) -> None:
        # use xavier initialization
        self.weights = np.random.randn(no_of_inputs, no_of_neurons) * np.sqrt(1 / no_of_inputs)
        self.bias = np.zeros((1, no_of_neurons))
        self.activation = activation
        
    def forward(self, inputs):
        self.inputs = inputs
        # self.inputs represents the activations of the previous layer a^{(l)}    

        
        self.a = self.activation.activation(np.dot(self.inputs, self.weights) + self.bias) 
        self.f_prime_z = self.activation.activation_derivative(self.a)
        return self.a
      
    
    def backward(self, delta, weights):
        # Element-wise multiplication for the derivative of the activation function
        
        # 1/m is already included in the gradient of the loss function
        delta = np.dot(delta, weights.T) * self.f_prime_z
                
        self.dj_dw = np.dot(self.inputs.T, delta)
        self.dj_db = np.sum(delta, axis=0, keepdims=True)
        
        
        return delta
    
    def update_weights(self, learning_rate):
        # Update weights and bias
        self.weights -= learning_rate * self.dj_dw
        self.bias -= learning_rate * self.dj_db
    
class InputLayer:
    def __init__(self, input_size) -> None: # input_size is unnecessary here
        self.f_prime_z = None
        self.weights = None
    def forward(self, inputs):
        # Forward pass for the input layer is just passing the input data
        return inputs

    def backward(self, delta):
        # The input layer has no weights, so the gradient is not modified
        return delta
    def update_weights(self, learning_rate):
        pass

class OutputLayer:
    def __init__(self, no_of_inputs, no_of_neurons, activation) -> None:
        self.weights = np.random.randn(no_of_inputs, no_of_neurons) * np.sqrt(1 / no_of_inputs)
        self.bias = np.zeros((1, no_of_neurons))
        self.f_prime_z = None
        self.activation = activation
        # ic(activation)
        
        # ic(self.weights.shape)
        
        # if no_of_neurons == 1:
        #     self.activation = Identity()
        # else:
        #     self.activation = Softmax()
            
    def forward(self, inputs):
        self.inputs = inputs
        # self.z = np.dot(self.inputs, self.weights)
        # self.a = self.activation.activation(np.dot(self.inputs, self.weights))
        self.a = self.activation.activation(np.dot(self.inputs, self.weights)+self.bias)
        return self.a
    
    def backward(self, delta):
        # grad_output is the gradient of the loss function with respect to the output of the layer
        # The gradient of the loss with respect to the input of the layer
        
        # self.dzdW = inputs.T
        # basically self.inputs.T is the same as self.dzdW
        self.dj_dw = np.dot(self.inputs.T, delta)
        self.dj_db = np.sum(delta, axis=0, keepdims=True)
        
        # grad_weights = np.dot(self.inputs.T, grad_output)
        # grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # The gradients of the loss with respect to the parameters of the layer
        
        # delta last layer
        return delta
    
    def update_weights(self, learning_rate):
        # Update weights and bias
        self.weights -= learning_rate * self.dj_dw
        self.bias -= learning_rate * self.dj_db
    
    # update will look something like this
    # def update(self, learning_rate):
    #     # Update weights and bias
    #     self.weights -= learning_rate * dj_dw
    #     self.bias -= learning_rate * self.grad_bias
    
class MLP:
    def __init__(self, input_size, hidden_layer_sizes, output_size,\
                 activation_function, output_activation_function, optimizer, loss, learning_rate = 0.001) -> None:
        """
        input_size: number of features in the input
        
        hidden_layer_sizes: list of number of neurons in each hidden layer
        
        output_size: number of classes in the output
        
        activation_function: list of activation function for each layer
        
        learning_rate: learning rate for gradient 
        """
        
        self.act_dict = {
            "sigmoid": Sigmoid(),
            "tanh": Tanh(),
            "relu": ReLU(),
            "identity": Identity(),
            "softmax": Softmax()
        }
        
        
        # # initial plan was to implement classes for the optimizers
        # self.opt_dict = {
        #     "sgd": SGD(),
        #     "bgd": BGD(),
        #     "mbsgd" : MBGD()
        # }
        
                
        for act in activation_function:
            act = act.lower()
            
        self.optimizer = optimizer
        self.output_activation_function = output_activation_function
        
        self.layers = []
        # self.activation_function = []
        self.learning_rate = learning_rate
        self.loss = loss
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.train_rmse = []
        self.val_rmse = []
        self.train_mse = []
        self.val_mse = []
        self.train_r2_score = []
        self.val_r2_score = []
        
        # self.y_pred = None

        # initialize the input layer
        self.layers.append(InputLayer(input_size))
        
        hidden_layer_sizes.insert(0, input_size)
        # +1 for the ones we are inserting in X
        activation_function.insert(0, None)
        # ic(hidden_layer_sizes)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function
        
        for i in range(1, len(hidden_layer_sizes)):
            # ic(i)
            self.layers.append(Layer(no_of_inputs=hidden_layer_sizes[i-1],no_of_neurons= hidden_layer_sizes[i], activation=self.act_dict[activation_function[i]]))

        self.layers.append(OutputLayer(hidden_layer_sizes[-1], output_size, self.act_dict[output_activation_function]))
        # print(len(self.layers))
    
    def forward_propagation(self, x)-> None:
        # Forward propagation
        for layer in self.layers:
            x = layer.forward(x)
        return x
        # return Softmax().activation(x)
        # return x
    
    # def backward_propagation(self, x, y)-> None:
    
    def backward_propagation(self, y_pred, y_true)-> None:
        # Forward propagation
        # self.forward_propagation(x)
        
        # calculate the loss gradient with respect to the output of the last layer
        # which is basically dj_dz for the output layer
        # delta_L represents delta for the last layer
        
        delta_L = self.loss.gradient(y_pred, y_true)
        # print(delta_L)
        # grad is dj_dz for the output layer
        # Backpropagation
        delta = delta_L
        # for layer in reversed(self.layers):
        #     delta = layer.backward(delta)
        # do the backword propagation for output layer explicitly and then the hidden layers and then for the input layer
        delta = self.layers[-1].backward(delta)
        # do the backword propagation for hidden layers
        for i in range(len(self.layers) - 2, 0, -1):
            delta = self.layers[i].backward(delta, self.layers[i+1].weights)
        # do the backword propagation for input layer
        delta = self.layers[0].backward(delta)
    
    def predict(self, x):
        # Forward propagation
        return self.forward_propagation(x)
    
    def update_weights(self):
        for layer in self.layers:
            layer.update_weights(self.learning_rate)
        
    
    # Working version of train
    # def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
    #     self.train_losses = []
    #     for epoch in range(epochs):
    #         loss = []
    #         for i in range(0, X_train.shape[0], batch_size):
    #             X_batch = X_train[i:i+batch_size]
    #             y_batch = y_train[i:i+batch_size]
    #             y_pred = self.forward_propagation(X_batch)
    #             self.backward_propagation(y_pred=y_pred, y_true=y_batch)
    #             self.update_weights()
    #             # calculate the loss
    #             loss.append(self.loss.loss(y_pred, y_batch))
    #         self.train_losses.append(np.mean(loss))
    
    def fit(self, X_train, y_train, epochs, X_val=None, y_val=None, batch_size=None):
        if self.optimizer == "bgd":
            self.train_bgd(X_train, y_train, X_val, y_val, epochs)
        if self.optimizer == "sgd":
            self.train_sgd(X_train, y_train, X_val, y_val, epochs)
        if self.optimizer == "mbgd":            
            self.train_mbgd(X_train, y_train, X_val, y_val, epochs, batch_size)
            
    def accuracy_helper(self, y_pred, y_true):
        return accuracy_score(y_true = np.argmax(y_true , axis=1), y_pred= np.argmax(y_pred, axis=1))
        
            
    def train_sgd(self, X_train, y_train, X_val, y_val, epochs):
        for epoch in range(epochs):
            # generate a random index
            for i in range(X_train.shape[0]):
                k = np.random.randint(0, X_train.shape[0])
                X_batch = X_train[k:k+1]
                y_batch = y_train[k:k+1]
                y_pred = self.forward_propagation(X_batch)
                self.backward_propagation(y_pred=y_pred, y_true=y_batch)
                self.update_weights()
            # calculate the loss for the entire training set
            y_pred_train = self.forward_propagation(X_train)
            
            loss_train = self.loss.loss(y_pred_train, y_train)
            self.train_losses.append(loss_train)
            if self.output_activation_function == "softmax":
                acc_train = self.accuracy_helper(y_pred_train, y_train)
                self.train_accuracies.append(acc_train)
            elif self.output_activation_function == "identity":
                # calculate RMSE MSE and R-Square
                self.train_mse(mean_squared_error(y_train, y_pred_train))
                self.train_rmse(np.sqrt(mean_squared_error(y_train, y_pred_train)))
                self.train_r2_score(r2_score(y_train, y_pred_train))
            
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward_propagation(X_val)
                
                loss_val = self.loss.loss(y_pred_val, y_val)
                self.val_losses.append(loss_val)
                if self.output_activation_function == "softmax":
                    val_acc = self.accuracy_helper(y_pred_val, y_val)
                    self.val_accuracies.append(val_acc)
                elif self.output_activation_function == "identity":
                    self.val_mse(mean_squared_error(y_val, y_pred_val))
                    self.val_rmse(np.sqrt(mean_squared_error(y_val, y_pred_val)))
                    self.vaval_r2_score(r2_score(y_val, y_pred_val))
                    
        
            
    def train_bgd(self, X_train, y_train, X_val, y_val, epochs):
        for epoch in range(epochs):
            y_pred = self.forward_propagation(X_train)
            self.backward_propagation(y_pred=y_pred, y_true=y_train)
            self.update_weights()
            # calculate the loss for the entire training set
            y_pred_train = self.forward_propagation(X_train)
            
            loss_train = self.loss.loss(y_pred_train, y_train)
            self.train_losses.append(loss_train)
            if self.output_activation_function == "softmax":
                train_acc = self.accuracy_helper(y_pred_train, y_train)
                self.train_accuracies.append(train_acc)
            elif self.output_activation_function == "identity":
                # calculate RMSE MSE and R-Square
                self.train_mse(mean_squared_error(y_train, y_pred_train))
                self.train_rmse(np.sqrt(mean_squared_error(y_train, y_pred_train)))
                self.train_r2_score(r2_score(y_train, y_pred_train))
            
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward_propagation(X_val)
                
                loss_val = self.loss.loss(y_pred_val, y_val)
                self.val_losses.append(loss_val)
                if self.output_activation_function == "softmax":
                    val_acc = self.accuracy_helper(y_pred_val, y_val)
                    self.val_accuracies.append(val_acc)
                elif self.output_activation_function == "identity":
                    self.val_mse(mean_squared_error(y_val, y_pred_val))
                    self.val_rmse(np.sqrt(mean_squared_error(y_val, y_pred_val)))
                    self.val_r2_score(r2_score(y_val, y_pred_val))
            
    
    def train_mbgd(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                y_pred = self.forward_propagation(X_batch)
                self.backward_propagation(y_pred=y_pred, y_true=y_batch)
                self.update_weights()
            # calculate the loss for the entire training set
            y_pred_train = self.forward_propagation(X_train)
            
            loss_train = self.loss.loss(y_pred_train, y_train)
            self.train_losses.append(loss_train)
            if self.output_activation_function == "softmax":
                train_acc = self.accuracy_helper(y_pred_train, y_train)
                self.train_accuracies.append(train_acc)
            elif self.output_activation_function == "identity":
                # calculate RMSE MSE and R-Square
                self.train_mse(mean_squared_error(y_train, y_pred_train))
                self.train_rmse(np.sqrt(mean_squared_error(y_train, y_pred_train)))
                self.train_r2_score(r2_score(y_train, y_pred_train))
            
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward_propagation(X_val)
                loss_val = self.loss.loss(y_pred_val, y_val)
                self.val_losses.append(loss_val)
                if self.output_activation_function == "softmax":
                    val_acc = self.accuracy_helper(y_pred_val, y_val)
                    self.val_accuracies.append(val_acc)           
                elif self.output_activation_function == "identity":
                    self.val_mse(mean_squared_error(y_val, y_pred_val))
                    self.val_rmse(np.sqrt(mean_squared_error(y_val, y_pred_val)))
                    self.val_r2_score(r2_score(y_val, y_pred_val))
            
            
