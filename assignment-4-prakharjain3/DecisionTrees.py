from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, X_train, y_train, epochs=None):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)