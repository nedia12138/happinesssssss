import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SuperiorLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lambda_reg=1e-5):
        self.lambda_reg = lambda_reg
        self.theta = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # 转换数据格式
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]

        XTX = X_b.T @ X_b
        # 正则化项
        reg_matrix = self.lambda_reg * np.eye(XTX.shape[0])
        reg_matrix[0, 0] = 0

        try:
            self.theta = np.linalg.solve(XTX + reg_matrix, X_b.T @ y)
        except np.linalg.LinAlgError:
            self.theta = np.linalg.pinv(XTX + reg_matrix) @ X_b.T @ y

        self.theta = self.theta.flatten()
        self.intercept_ = float(self.theta[0])
        self.coef_ = self.theta[1:]
        return self

    def predict(self, X):
        X = np.array(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y).flatten()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)