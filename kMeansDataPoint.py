import numpy as np

# A data point class to use for the clustering algorithm
class kDataPoint:

    def __init__ (self, data , windowSize, day):
            # the marketVariance for the current window
            self.marketVar = np.var(data)
            # the marketMean return for the current window
            self.marketMean = np.mean(data)
            # an array of each stocks individual variance for a given window
            if windowSize > 1:
                self.stockVars = np.var(data, axis = 1)
            else:
                self.stockVars = np.zeros(len(data))
            # an array of each stocks individual variance for a given window
            if windowSize > 1:
                self.stockMeans = np.mean(data, axis = 1)
            else:
                self.stockMeans = data
            # id of the cluster should be between 0 and numClusters - 1
            self.kCluster = -1
            # setUpTheData
            self.windowSize = windowSize
            # need the date to check back to start
            # this will help with back tracking
            self.day = day

    def calcMarketVar(self,data):
        self.marketVar = np.var(data)
        print(np.var(data))

    def calcMarkMean(self, data): 
        self.marketMean = np.mean(data)

    def calcStockVar(self,data):
        self.stockVars = np.var(data, axis = 1)

    def calcStockMean(self,data):
        self.stockMeans = np.mean(data, axis = 1)

    def setUpData(self, data):
        self.marketVar = self.calcMarketVar(data)
        self.marketMean = self.calcMarkMean(data)
        self.stockVars = self.calcStockVar(data)
        self.stockMeans = self.calcStockMean(data)
        print("I set up")
    
    def calcDist(self, cluster):
        if cluster.marketMean is not None:
            try:
                distMarketMean = np.abs(self.marketMean - cluster.marketMean)
                distMarketVar = np.abs(self.marketVar - cluster.marketVar)
                tempMean = np.abs(self.stockMeans - cluster.stockMeans)
                tempVar = np.abs(self.stockVars - cluster.stockVars)
                dist = distMarketMean + distMarketVar + np.sum(tempMean) + np.sum(tempVar)
                # dist = np.sqrt(dist)
                return dist
            except:
                return 1e+9
        else:
            return 1e+9

    def reassignCluster(self, clusters):
        # Distance vastly too large so we can find a good minimum distance
        minDist = 1e+8
        # assume we are passing in a list of clusters
        bestCluster = -1
        for i in clusters:
            dist = self.calcDist(i)
            # print("Distance: " + str(dist) + " , minDist: " + str(minDist))
            if dist < minDist:
                minDist = dist
                bestCluster = int(i.id)
        self.kCluster = bestCluster
        return

    def printPoint(self):
        print("K Means Data Point, following att:\n")
        print("MarketVariance: " + str(self.marketVar))
        print("MarketMean: " + str(self.marketMean))
        print("StockMean: " + str(self.stockMeans))
        print("StockVariance: " + str(self.stockMeans))

