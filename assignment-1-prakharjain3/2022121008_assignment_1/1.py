import sys
import numpy as np
from KNN import KNNClassifier
from sklearn.metrics import classification_report
import pandas as pd
from tabulate import tabulate


# Print the script name
if(len(sys.argv) < 2):
    print("Usage: python3 1.py <file_name_or_path or path> <k> <distance_metric> here file_name_or_path is required")
    exit()

def get_args_for_knn():

    distance_metric = 'euclidean'
    k = 3
    file_name_or_path = None
    for arg in sys.argv[1:]:
        if arg in ['euclidean', 'cosine', 'manhattan', 'minkowski']:
            distance_metric = arg
        elif arg.isdigit():
            k = int(arg)
        else:
            file_name_or_path = arg
    return file_name_or_path, k, distance_metric

file_name_or_path, k, distance_metric = get_args_for_knn()

# print("file_name_or_path: ", file_name_or_path)
# print("k: ", k)
# print("distance_metric: ", distance_metric)
# print("encoder: ", encoder)

data = np.load(file_name_or_path, allow_pickle=True)

y_test = data[:, 3]

knn = KNNClassifier(k=k,distance_metric=distance_metric, encoder='ResNet')
X_test = data[:, 1]
y_pred = knn.predict(X_test)
print("\u001b[1mResNet\u001b[0m")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

df = pd.DataFrame(report).transpose()


# # print(df.to_markdown())
print(tabulate(df, headers='keys', tablefmt='psql'))


X_test = data[:, 2]
knn = KNNClassifier(k=k,distance_metric=distance_metric, encoder='VIT')
y_pred = knn.predict(X_test)
print("\u001b[1mVIT\u001b[0m")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

df = pd.DataFrame(report).transpose()
print(tabulate(df, headers='keys', tablefmt='psql'))