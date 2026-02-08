import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


# --- 第一部分：手写回归决策树 ---
class MyDecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        # 这里的 X 是 numpy 数组
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        # 停止条件
        if depth >= self.max_depth or num_samples < self.min_samples_split or np.std(y) == 0:
            return np.mean(y)

        # 随机森林精髓：在每个分裂点只随机选一部分特征（sqrt(n)个）
        feat_idxs = np.random.choice(num_features, max(1, int(np.sqrt(num_features))), replace=False)

        best_feat, best_thresh, best_var = None, None, float('inf')

        for feat_idx in feat_idxs:
            # 性能优化：只尝试 10 个百分位点作为阈值，而不是尝试所有数值
            thresholds = np.percentile(X[:, feat_idx], [10, 20, 30, 40, 50, 60, 70, 80, 90])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2: continue

                # 计算加权方差
                var = np.var(y[left_mask]) * np.sum(left_mask) + np.var(y[right_mask]) * np.sum(right_mask)
                if var < best_var:
                    best_feat, best_thresh, best_var = feat_idx, thresh, var

        if best_feat is None: return np.mean(y)

        # 递归构建
        left = self._build_tree(X[X[:, best_feat] <= best_thresh], y[X[:, best_feat] <= best_thresh], depth + 1)
        right = self._build_tree(X[X[:, best_feat] > best_thresh], y[X[:, best_feat] > best_thresh], depth + 1)
        return {'feat': best_feat, 'thresh': best_thresh, 'left': left, 'right': right}

    def predict(self, X):
        return np.array([self._traverse(x, self.tree) for x in X])

    def _traverse(self, x, node):
        if not isinstance(node, dict): return node
        if x[node['feat']] <= node['thresh']:
            return self._traverse(x, node['left'])
        return self._traverse(x, node['right'])


# --- 第二部分：手写随机森林 ---
class SuperiorRandomForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        X, y = np.array(X), np.array(y)
        for _ in range(self.n_estimators):
            # Bootstrap 抽样（自助采样）
            idxs = np.random.choice(len(X), len(X), replace=True)
            tree = MyDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        # 森林预测结果是所有树的平均值
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)