import numpy as np
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean', encoder="VIT"):
        self.k = k
        self.distance_metric = distance_metric
        self.encoder = encoder

        file = "data.npy"
        data = np.load(file, allow_pickle=True)

        resnet_index = 1
        vit_index = 2
        label_index = 3

        y = data[:, label_index]

        if self.encoder == "ResNet":

            X_resnet = []
            for d in data[:, resnet_index]:
                X_resnet.append(d.reshape(-1))
            X_resnet = np.array(X_resnet)
            X_resnet_train, X_resnet_test, y_resnet_train, y_resnet_test = train_test_split(X_resnet, y, test_size=0.2, random_state=2)

            self.X_train = X_resnet_train
            self.y_train = y_resnet_train
            self.X_test = X_resnet_test
            self.y_test = y_resnet_test

        elif self.encoder == "VIT":

            X_vit = []
            for d in data[:, vit_index]:
                X_vit.append(d.reshape(-1))
            X_vit = np.array(X_vit)
            X_vit_train, X_vit_test, y_vit_train, y_vit_test = train_test_split(X_vit, y, test_size=0.2, random_state=2)

            self.X_train = X_vit_train
            self.y_train = y_vit_train
            self.X_test = X_vit_test
            self.y_test = y_vit_test

        self.min = self.X_train.min(axis=0)
        self.max = self.X_train.max(axis=0)

        # self.mean = self.X_train.mean(axis=0)
        # self.std = self.X_train.std(axis=0)

        self.X_train = self.min_max_normalization(self.X_train)

    def min_max_normalization(self, x):
        return (x - self.min) / (self.max - self.min)

    def euclidean_distance(self, x):
        return np.sqrt(np.sum((self.X_train - x)**2, axis=1))
    def manhattan_distance(self, x):
        return np.sum(np.abs(self.X_train - x), axis=1)
    def cosine_distance(self, x):
        return (1 - (np.sum(self.X_train * x, axis=1) / (np.sqrt(np.sum(self.X_train**2, axis=1)) * np.sqrt(np.sum(x**2)))))
    def minkowski_distance(self, x, p=3):
        return np.sum(np.abs(self.X_train - x)**p, axis=1)**(1/p)
    
    def predict(self, X):
        predictions = []
        # inference_time = []

        for x in X:
            x = self.min_max_normalization(x)

            # start_time = time.time()

            prediction = self.real_predict(x)
            # inference_time.append(time.time() - start_time)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def real_predict(self, x):
        distances = np.zeros(self.X_train.shape[0])
        if self.distance_metric == 'euclidean':
            distances = self.euclidean_distance(x)
        elif self.distance_metric == 'cosine':
            distances = self.cosine_distance(x)
        elif self.distance_metric == 'manhattan':
            distances = self.manhattan_distance(x)
        elif self.distance_metric == 'minkowski':
            distances = self.minkowski_distance(x)

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        k_nearest_distances = distances[k_indices]

        unique_labels = np.unique(k_nearest_labels)
        
        # Calculate weights based on inverse distances
        weights = 1 / (k_nearest_distances + 1e-10)  # Avoid division by zero
        
        # Create a dictionary to store the weighted counts for each label
        weighted_count_dict = {}
        for label in unique_labels:
            indices = np.where(k_nearest_labels == label)
            weighted_count_dict[label] = np.sum(weights[indices])
        # print(weighted_count_dict)
                
        # Find the label with the highest weighted count
        highest_weighted_label = max(weighted_count_dict, key=weighted_count_dict.get)
        
        return highest_weighted_label

    def get_inference_and_scores(self):
        
        start = time.time()
        prediction = self.predict(self.X_test)
        inference_time = time.time() - start

        dict = classification_report(y_pred=prediction, y_true=self.y_test, zero_division=0, output_dict=True)

        return {
            "prediction":prediction,
            "dict":dict,
            "inference_time":inference_time
                }
        