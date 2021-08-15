import numpy as np

class fuzzyCentroid:

    def __init__ (self):
        
        # initialise to all zeros to begin
        # add and remove as we continue but test out like this for now
        
        # the marketVariance for the current window
        self.marketVar = 0.0
        # the marketMean return for the current window
        self.marketMean = 0.0
        # an array of each stocks individual variance for a given window
        self.stockVars = None
        # an array of each stocks individual variance for a given window
        self.stockMean = None
        # id
        self.id = -1

    def setFuzzyCentroidID(self, id):
        # assocs will be how many centroids a given data point can be associated to
        self.id = id
        