import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


# --- 第一部分：手写回归决策树基类 ---
class MyDecisionTreeRegressor:
    """
    底层回归决策树：
    使用方差（Variance）作为分裂准则，通过递归构建二叉树来拟合数据规律。
    """

    def __init__(self, max_depth=5, min_samples_split=10, max_features='all'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # 特征抽样策略，决定分裂时看多少个特征
        self.tree = None

    def fit(self, X, y):
        # 启动递归构建流程
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        # 【停止条件】
        # 1. 达到最大深度（防止过拟合）
        # 2. 节点样本数太少（不足以继续分裂）
        # 3. 目标变量标准差为0（样本完全一致，无需分裂）
        if depth >= self.max_depth or num_samples < self.min_samples_split or np.std(y) == 0:
            return np.mean(y)

        # 【特征子集选择 - 方案选择】
        # 随机森林精髓：在每个分裂点只随机选一部分特征，增加树的多样性
        if self.max_features == 'sqrt':
            n_select = max(1, int(np.sqrt(num_features)))  # 分类任务常用
        elif self.max_features == 'all':
            n_select = num_features  # 方案A：使用全部特征，减少回归任务的随机误差
        elif isinstance(self.max_features, float):
            n_select = max(1, int(self.max_features * num_features))  # 自定义比例
        else:
            n_select = max(1, int(num_features / 3))  # 方案B：回归任务常用(1/3特征)

        # 执行无放回随机抽样，选出本次分裂要考虑的特征索引
        feat_idxs = np.random.choice(num_features, n_select, replace=False)

        best_feat, best_thresh, best_var = None, None, float('inf')

        # 【寻找最佳分裂点】
        for feat_idx in feat_idxs:
            # 性能优化：只尝试 10 个百分位点作为阈值，避免全样本扫描，显著提升大数据训练速度
            thresholds = np.percentile(X[:, feat_idx], [10, 20, 30, 40, 50, 60, 70, 80, 90])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask

                # 确保分裂后的子集至少有2个样本，保证方差计算有意义
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue

                # 计算加权方差总和：目标是让分裂后的两个子集内部尽可能“纯”（方差小）
                var = np.var(y[left_mask]) * np.sum(left_mask) + np.var(y[right_mask]) * np.sum(right_mask)
                if var < best_var:
                    best_feat, best_thresh, best_var = feat_idx, thresh, var

        # 如果在该节点找不到有效分裂点，则退化为叶子节点，返回均值
        if best_feat is None:
            return np.mean(y)

        # 【递归构建】
        # 根据最佳特征和阈值将数据切分为左右两块，继续向下生长
        left = self._build_tree(X[X[:, best_feat] <= best_thresh], y[X[:, best_feat] <= best_thresh], depth + 1)
        right = self._build_tree(X[X[:, best_feat] > best_thresh], y[X[:, best_feat] > best_thresh], depth + 1)

        return {'feat': best_feat, 'thresh': best_thresh, 'left': left, 'right': right}

    def predict(self, X):
        # 对输入矩阵中的每一行数据，沿着树结构向下寻找对应的叶子节点
        return np.array([self._traverse(x, self.tree) for x in X])

    def _traverse(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feat']] <= node['thresh']:
            return self._traverse(x, node['left'])
        return self._traverse(x, node['right'])


# --- 第二部分：手写随机森林集成器 ---
class SuperiorRandomForest(BaseEstimator, RegressorMixin):
    """
    基于 Bootstrap 抽样和多棵回归树的集成学习模型：
    通过“众筹决策”的思想，取多棵树的平均值作为最终预测结果，有效降低单棵树的过拟合风险。
    """

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=20, max_features='all'):
        self.n_estimators = n_estimators  # 森林中树的数量
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        X, y = np.array(X), np.array(y)
        for _ in range(self.n_estimators):
            # 【Bootstrap 抽样】
            # 有放回地随机抽取样本，使每一棵树看到的数据集都有微小差异，增加森林的鲁棒性
            idxs = np.random.choice(len(X), len(X), replace=True)

            # 独立初始化并训练每一棵决策树
            tree = MyDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        # 【结果聚合 - Aggregation】
        # 森林的最终输出是所有决策树预测值的简单平均（Mean Pooling）
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)