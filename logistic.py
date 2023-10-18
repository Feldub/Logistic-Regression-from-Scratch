import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + np.e**(-t))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            row = np.concatenate([[1], row])
        t = np.dot(row, coef_)
        return self.sigmoid(t)


if __name__ == '__main__':
    model = CustomLogisticRegression()
    coef = np.array([0.77001597, -2.12842434, -2.39305793])
    results = []
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X[['worst concave points', 'worst perimeter']]
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
    for _, values in X_test.iloc[:10].iterrows():
        results.append(model.predict_proba(values, coef))
    print(results)


