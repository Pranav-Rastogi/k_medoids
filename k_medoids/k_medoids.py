import numpy as np

def _mDistance(start,end):
    return sum(abs(e - s) for s,e in zip(start,end))
    
def _random(bound,size):
    _rv = []
    _vis = []
    while True:
        r = np.random.randint(bound)
        if r in _vis:
            pass
        else:
            _vis.append(r)
            _rv.append(r)
        
        if len(_rv) == size:
            return _rv
    
class K_medoids:
    
    def __init__(self,n_clusters=2,max_iters=100):
        
        if n_clusters < 2:
            raise ValueError("Value of 'n_clusters' cannot be less than 2")
        self.k = n_clusters
        self.max_iters = max_iters
        self._fit = False
    
    def fit(self,data):
        
        self.medoids = {}
        self.clusters = {}
        
        _med = _random(len(data),self.k)
        
        for i in range(self.k):
            self.medoids[i] = data[_med[i]]
        
        for _ in range(self.max_iters):
            for j in range(self.k):
                self.clusters[j] = []
                
            for point in data:
                distances = [_mDistance(point, self.medoids[i]) if point.all() == self.medoids[i].all() else 0  for i in self.medoids]
                min_dist = np.min(distances)
                index = distances.index(min_dist)
                self.clusters[index].append(point)
                
            for i in self.clusters:
                distances = [[_mDistance(point, every_point) if point.all() == every_point.all() else 0 for every_point in self.clusters[i]] for point in self.clusters[i]] 
                costs = list(np.sum(distances, axis=1))
                index = costs.index(np.min(costs))
                self.medoids[i] = self.clusters[i][index]
    
        self._fit = True
                
                
    def predict(self, data):
        if self._fit == False:
            raise RuntimeError('a program call to fit() must be given before calling predict()')
            
        predictions = []
        for point in data:
            distances = [_mDistance(point, self.medoids[i]) for i in self.medoids]
            min_dist = np.min(distances)
            index = distances.index(min_dist)
            predictions.append(index)
            
        return predictions        
     
