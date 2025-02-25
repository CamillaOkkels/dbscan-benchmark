from benchmark.algorithms.base.module import BaseDBSCAN

import dbscan_srr as dbscan
import numpy as np

class SRRDBSCAN(BaseDBSCAN):
    def __init__(self, delta, memory, threads, shrinkage, approxFactor):
        self.delta = delta
        self.memory = memory
        self.threads = threads
        self.shrinkage = shrinkage # only look at shrinkage * 100% of repetitions
        self.approx = approxFactor
        self.srr = dbscan.SRR()

    def cluster(self, X: np.array, epsilon: float, minPts:int): 
        self.clustering = self.srr.fit_predict(X, self.delta, self.memory, True, "test", self.threads, -1, self.shrinkage, epsilon, minPts, self.approx)

    def retrieve_labels(self):
        self.clustering = np.array(self.clustering)
        return self.clustering, self.clustering[self.clustering > -1], [] 

    def get_additional(self):
        return self.srr.statistics()
    
    def __str__(self):
        return f"SRRDBSCAN(delta={self.delta}, memory={self.memory}, threads={self.threads}, shrinkage={self.shrinkage}, approx={self.approx})"

    def __repr__(self):
        return f"{self.delta}_{self.memory}_{self.threads}_{self.shrinkage}_{self.approx}"
