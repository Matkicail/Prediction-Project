import numpy as np

from kMeansClustering import kCentroid

from kMeansDataPoint import kDataPoint

def createPoints(trainSize, windowSize, marketData):
    """
    A function that creates data points using a given window size and a market of price relatives.
    """
    dataPoints = []
    for i in range(windowSize,trainSize):
        data = marketData[:,i-windowSize:i]
        point = kDataPoint(data, windowSize, i)
        dataPoints.append(point)
    return dataPoints

def generateCentroids(numCluster):
    """
    Generates the centroids given a number of centroids.
    """
    centroids = []
    for i in range(numCluster):
        centroid = kCentroid()
        centroid.id = i
        centroids.append(centroid)
    return centroids

def randomAssignCentroids(dataPoints, numCluster):
    """
    Assigns a random cluster to each data point for the initialisation process
    """
    assigned = np.random.randint(low=0, high=numCluster, size=len(dataPoints))
    count = 0
    for i in dataPoints:
        i.kCluster = assigned[count]
        count += 1

def updateCentroids(centroids, dataPoints, numCluster):
    """
    For all our centroids in the clustering algorithm, update their means based on the data points.
    """
    points = np.array(([]))
    for i in dataPoints:
        points = np.append(points, i.kCluster)
    for i in range(0,numCluster):
        temp = []
        valsToAdd = np.where(points == i)[0]
        if len(valsToAdd) > 0:
            for j in valsToAdd:
                temp.append(dataPoints[j])
            centroids[i].updateKCentroid(temp)
    return centroids

def updateClusters(centroids, dataPoints):
    """
    Updating all data points based on the centroids that are available.
    """
    before = currentAssignments(dataPoints)
    for i in dataPoints:
        i.reassignCluster(centroids)
    after = currentAssignments(dataPoints)
    diff = after - before
    noChange = len(np.where(diff != 0)[0])
    change = noChange / len(dataPoints)
    print("Percentage changed: " + str(change*100) + "%")
    return dataPoints, change

def currentAssignments(dataPoints):
    """
    Find the current assignments for each cluster to check what they are and find out what they do.
    """
    curr = np.array(())
    for i in range(len(dataPoints)):
        curr = np.append(curr, dataPoints[i].kCluster)
    return curr      

def reAdjustDataAssign(centroids, dataPoints, numCluster, tol):
    """
    Check to see if we can stop the clustering algorithm since we are within the acceptable tolerance.
    """
    currError = 1
    while currError > tol:
        centroids = updateCentroids(centroids, dataPoints, numCluster)
        dataPoints, currError = updateClusters(centroids, dataPoints)

trainSize = 5000
marketData = np.loadtxt("./Data Sets/PriceRelatives/BOVPRICERELATIVES.txt")
initialSegment = marketData[:,0:trainSize]

# First hyper parameters - number of clusters & a window size
numCluster = int(50)
print("The number of clusters is: " + str(numCluster))
windowSize = 5
tol = 1e-3

# create points that look from today up until the final day - train size
dataPoints = createPoints(trainSize, windowSize, marketData)
centroids = generateCentroids(numCluster)
randomAssignCentroids(dataPoints, numCluster)
reAdjustDataAssign(centroids, dataPoints, numCluster, tol)


