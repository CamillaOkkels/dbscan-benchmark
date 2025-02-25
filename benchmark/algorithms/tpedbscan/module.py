from benchmark.algorithms.base.module import BaseDBSCAN
from dbscan import DBSCAN

import numpy as np

class TPEDBSCAN(BaseDBSCAN):
    def __init__(self):
        pass

    def cluster(self, X: np.array, epsilon: float, minPts:int):  
        self.labels, isCore = DBSCAN(X, eps=epsilon, min_samples=minPts)
        self.core_points = np.array((np.array(list(range(X.shape[0])))[isCore]))

    def retrieve_labels(self):
        return self.labels, self.core_points, []

    def __str__(self):
        return f"TPEDBSCAN()"

    def __repr__(self):
        return f"run"
