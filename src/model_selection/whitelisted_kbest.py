from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression, SelectKBest
import numpy as np

class WhitelistedSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k, whitelist_idx, score_func=f_regression):
        self.whitelist_idx = whitelist_idx

        self.k = k
        self.score_func = score_func
        self.selector = SelectKBest(score_func=score_func, k=k)

        self.scores_ = None
        self.pvalues_ = None

    def fit(self, X, y):
        self.selector.fit(X, y)
        selected_idx = self.selector.get_support(indices=True)

        # combine whitelisted and selected features
        all_idx = list(set(self.whitelist_idx) | set(selected_idx))

        # update the selector mask to include the whitelisted features
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[all_idx] = True

        # update the support function to return the entire mask on default call or the selected indices on indices=True
        self.selector.get_support = lambda indices = False : mask if not indices else np.where(mask)[0]

        self.scores_ = self.selector.scores_
        self.pvalues_ = self.selector.pvalues_

        return self

    def transform(self, x):
        return self.selector.transform(x)

    def get_support(self, indices=False):
        return self.selector.get_support(indices=indices)