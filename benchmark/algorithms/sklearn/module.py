from benchmark.algorithms.base.module import BaseDBSCAN
from sklearn.cluster import DBSCAN

import numpy as np

class SKLearnDBSCAN(BaseDBSCAN):
    def __init__(self):
        pass

    def cluster(self, X: np.array, epsilon: float, minPts:int):  
        self.clustering = DBSCAN(eps=epsilon, min_samples=minPts, algorithm="ball_tree").fit(X)

    def retrieve_labels(self):
        return self.clustering.labels_, self.clustering.core_sample_indices_, []
    
    def __str__(self):
        return f"SKLEARNDBSCAN()"

    def __repr__(self):
        return f"run"