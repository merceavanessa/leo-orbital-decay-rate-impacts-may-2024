import os
from sklearn.model_selection import BaseCrossValidator
import numpy as np

class FixedRollingSplit(BaseCrossValidator):
    def __init__(self, train_size, n_splits=None, step=None, test_size = None, fixed_test_to_end=False, start=0):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.step = step or test_size
        self.fixed_test_to_end = fixed_test_to_end
        self.start = start

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        start = self.start
        if self.n_splits is None:
            self.n_splits = n_samples // self.step

        if self.test_size is None:
            self.test_size = n_samples - self.train_size - start

        if start + self.train_size + self.test_size > n_samples:
            raise ValueError(f"Train size + test size + start must be less than the number of samples ({n_samples}). train : {self.train_size}, test : {self.test_size}, start : {start}, n_samples : {n_samples})")

        actual_n_splits = 0
        for i in range(self.n_splits):
            if start + self.train_size > n_samples:
                actual_n_splits = i
                break

            train_start = start
            train_end = train_start + self.train_size

            if not self.fixed_test_to_end:
                test_start = train_end
                test_end = test_start + self.test_size
            else:
                test_start = n_samples - self.test_size
                test_end = n_samples

            if (test_start < train_end) or (test_end > n_samples):
                actual_n_splits = i
                break

            yield np.arange(train_start, train_end), np.arange(test_start, test_end)
            start += self.step
            actual_n_splits += 1

        self.n_splits = actual_n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits