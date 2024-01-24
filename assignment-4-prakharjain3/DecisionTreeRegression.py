from sklearn.tree import DecisionTreeRegressor

class DecisionTreeRegressorWrapper:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    def fit(self, X_train, y_train, epochs = None):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_feature_importance(self):
        return self.model.feature_importances_

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)