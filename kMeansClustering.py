import numpy as np

class kCentroid:

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
        self.stockMeans = None
        # id of a cluster should be between 0 and numCluster-1
        self.id = -1
        # for simplicity later
        self.alive = True

    def setKCentroidID(self, id):
        # assocs will be how many centroids a given data point can be associated to
        self.id = int(id)
    
    def updateKCentroid(self, assignedCentroids):

        if len(assignedCentroids) > 1:
            centoridMarketVars = np.array(())
            centroidMarketMeans = np.array(())
            centroidStockVars = np.empty((assignedCentroids[0].stockVars.shape[0], len(assignedCentroids)))
            centroidStockMeans = np.empty((assignedCentroids[0].stockVars.shape[0], len(assignedCentroids)))
            count = 0
            for i in assignedCentroids:
                centoridMarketVars = np.append(centoridMarketVars, i.marketVar)
                centroidMarketMeans = np.append(centroidMarketMeans, i.marketMean)
                centroidStockVars[:,count] = i.stockVars
                centroidStockMeans[:,count] = i.stockMeans.flatten()
                count += 1
            self.marketVar = np.mean(centoridMarketVars)
            self.marketMean = np.mean(centroidMarketMeans)
            self.stockVars = np.mean(centroidStockVars, axis = 1)
            self.stockMeans = np.mean(centroidStockMeans, axis = 1)

        elif len(assignedCentroids) == 0:
            self.marketVar = assignedCentroids.marketVar
            self.marketMean = assignedCentroids.marketMean
            self.stockVars = assignedCentroids.stockVars
            self.stockMeans = assignedCentroids.stockMeans
        else:
            self.marketVar = -10
            self.marketMean = -10
            self.stockVars = -10
            self.stockMeans = -10
            self.alive = False

    def printCluster(self):
        print("K Means Centroid, following att:\n")
        print("MarketVariance: " + str(self.marketVar))
        print("MarketMean: " + str(self.marketMean))
        print("StockMean: " + str(self.stockMeans))
        print("StockVariance: " + str(self.stockVars))