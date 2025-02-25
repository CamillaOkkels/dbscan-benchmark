from benchmark.algorithms.base.module import BaseDBSCAN

import SubsampledNeighborhoodGraphDBSCAN as dbscan
import numpy as np

class SNGDBSCAN(BaseDBSCAN):
    def __init__(self, p):
        self.p = p

    def cluster(self, X: np.array, epsilon: float, minPts:int):  
        self.clustering = dbscan.SubsampledNeighborhoodGraphDBSCAN(eps=epsilon, minPts=minPts,p=self.p).fit_predict(X)

    def retrieve_labels(self):
        return self.clustering, [], []

    def __str__(self):
        return f"SNGDBSCAN(p={self.p})"

    def __repr__(self):
        return f"p_{self.p}"
