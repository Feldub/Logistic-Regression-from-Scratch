import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = None

    @staticmethod
    def add_bias(X):
        X_ = X.copy()
        bias_col = np.ones(X_.shape[0])
        X_.insert(0, 'bias', bias_col)
        return X_

    def sigmoid(self, t):
        return 1 / (1 + np.e**(-t))

    def predict_proba(self, row, coef_):
        t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        if self.fit_intercept:
            X_train = self.add_bias(X_train)

        self.coef_ = np.zeros(X_train.shape[1])
        for _ in range(self.n_epoch):
            for row_index, row in X_train.iterrows():
                y_hat = self.predict_proba(row, self.coef_)
                self.coef_ = (self.coef_ -
                              self.l_rate
                              * (y_hat - y_train[row_index])
                              * y_hat
                              * (1 - y_hat)
                              * row.values)

    def fit_log_loss(self, X_train, y_train):
        if self.fit_intercept:
            X_train = self.add_bias(X_train)

        self.coef_ = np.zeros(X_train.shape[1])
        for _ in range(self.n_epoch):
            for row_index, row in X_train.iterrows():
                y_hat = self.predict_proba(row, self.coef_)
                self.coef_ = (self.coef_
                              - self.l_rate
                              * (y_hat - y_train[row_index])
                              * row.values
                              / X_train.shape[0])

    def predict(self, X_test, cut_off=0.5):
        if self.fit_intercept:
            X_test = self.add_bias(X_test)
        predictions = []
        for _, row in X_test.iterrows():
            y_hat = self.predict_proba(row, self.coef_)
            predictions.append(int(y_hat >= cut_off))
        return np.array(predictions)


if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X[['worst concave points', 'worst perimeter', 'worst radius']]
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=43
    )
    lr = CustomLogisticRegression(
        fit_intercept=True, l_rate=0.01, n_epoch=1000
    )
    lr.fit_log_loss(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    print({
        'coef_': list(lr.coef_),
        'accuracy': acc_score
    })
