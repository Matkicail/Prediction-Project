from datetime import date
import numpy as np
from numpy.lib import corrcoef
import pandas as pd
from matplotlib import pyplot as plt
from stockMarketReader import readDataSet
import math
from scipy.optimize import minimize 
from kMeansClustering import kCentroid
from kMeansDataPoint import kDataPoint


# Need to construct a set of experts as required by CORN
class Expert:
    """
    This class serves the purpose of making a CORN expert
    """

    #constructor for this expert with the two noteworthy parameters
    def __init__(self, windowSize, corrThresh, numStocks, numDays):
        """
        Initialisation of a CORN expert with its unique parameters:
        the window size, the correlation threshold.
        Standard parameters being number of stocks and number of days.
        """
        self.windowSize = windowSize
        self.corrThresh = corrThresh
        self.numStocks = numStocks
        self.weight = 0
        # initial wealth as based on page 12 note
        self.wealthAchieved = 1
        self.currPort = None
        self.portHistory = np.empty((numStocks, numDays))
        
    def updateWeight(self, numExperts):
        """
        Update this agent's weight to be 1/numExperts as it is part of top-numExperts (K)
        """
        self.weight = 1 / numExperts

    def assignCorrSet(self, corrSet):
        """
        A function that allows us to add a correlation similar set to this specific expert - should it be needed.
        It will most likely not be needed given that this changes frequently.
        """
        self.corrSimSet = corrSet
    
    def addPort(self, portfolio, day):
        """
        A way to track past portfolios should it be needed.
        In reality this is not really going to be needed given that we can track the wealth and increase the wealth.
        """
        for i in range(self.numStocks):
            self.portHistory[day][i] = portfolio[i]
    
    def increaseWealth(self, priceVector):
        """
        Function that given a portfolio and a price relative vector will increase the agent's wealth using it.
        Note that this is meant to take in the day's (i.e at time t) price relative vector.
        """
        # need to set a self wealth for each specific agent
        self.wealthAchieved = self.wealthAchieved * (self.currPort @ priceVector)
    

def getUniformPort():
    """
    Generate a uniform portfolio given a number of stocks.
    """
    stocks = np.ones((numStocks))
    return stocks / numStocks

def objective(portfolio, days, setSize):
    """
    Ensuring that days is a vector/matrix where width is number of days and length is numStocks.
    """
    total = 1
    for i in range(days.shape[1]) :
        total *= portfolio @ days[:,i]
    # Return negative portfolio so that we can minimise (hence maximising the portfolio)
    return -total

def constraintSumOne(portfolio):
    prob = 1
    for i in portfolio:
        prob -= i
    return prob

def boundsCreator():
    b = (0.0,1.0)
    a = [b]*numStocks
    return a

def initialGuess(days, sizeSet):
    port = np.zeros((numStocks))
    for i in range(sizeSet):
        bestStock = np.argmax(days[:,i]) 
        port[bestStock] += 1
    port /= port.sum()
    
    return port

def expertLearn(window, corrThresh, day, data, dataPoints, Agent):
    """
    Preform algorithm 1 from CORN paper.
    This algorithm is the expert learning procedure.
    Given an index date (day), the window (a specified window size), the histMarketWind (t-1 to 1) and the corrThresh which is rho our correlation coeff threshold.
    """
    corrSimSet = np.array(())
    if day <= window + 1:
        #works
        return uniformPort
    else:
        #check out
        for i in dataPoints: #just check that this works otherwise change it to t
            # print("Test 1")
            # print("size of window: " + str(window))
            # print("I-window: " + str(i-window) + " , i-1: " + str(i-1))
            # print(data.shape)
            markWindI = marketWindow(i.day-window, i.day-1, dates, data)
            # print("Test 2")
            # print("day-window: " + str(day-window) + " , day-1: " + str(day-1))
            markWindT = marketWindow(day - window, day - 1, dates, data)
            # print("Test 3")
            # check at some point to ensure that this captures the standard deviation for the whole window (i.e output not something weird)
            # flattened just to ensure that this does happen
            pearsonCor = -1
            if np.std(markWindI.flatten()) == 0 or np.std(markWindT.flatten()) == 0:
                # print("Recognised a 0 standard deviation in a market window")
                pearsonCor = 0
            # print("Test 4")
            # may need to change this to the exact calculation they use in the formula
            if pearsonCor != 0:
                pearsonCor = np.cov(markWindI, markWindT) / (np.std(markWindI.flatten()) * np.std(markWindT.flatten()) )
                # print("Test 5")
            elif  pearsonCor >= corrThresh:
                # print("Appended a set")
                # append this to our set i.e add the index
                corrSimSet = np.append(corrSimSet,i.day)
            # print("Test 6")
    if len(corrSimSet) == 0 and window < 3:
        # print("Empty set")
        return uniformPort
    else:

        if len(corrSimSet) == 0:
            for i in dataPoints:
                corrSimSet = np.append(corrSimSet,i.day)
        else:
            # print(Agent)
            print("\t I found this many in my corrSimSet: " + str(len(corrSimSet)))
        # Search for the optimal portfolio
        # so using the i in the set, get the argmax
        # from what I understand, we need the price relative vector at time i, find the stock that gave the best return and all in on that stock

        # TODO ADD CHANGES BASED ON DRICORN-K (so here construct a portfolio based on their optimals)
        # do not need a temp relative just find the sub portfolios, log them, divide by the number in the corrSimSet
        # then minus the deviation
        # from here normalise the portfolio so that the required portfolio property is maintained
        tempRelative = 0
        port = np.zeros((numStocks))
        corrSimSetDays = np.empty((numStocks,len(corrSimSet)))
        # print(corrSimSetDays.shape)
        # print(numStocks)
        for i in range(len(corrSimSet)):
            corrSetDay = dayReturn(corrSimSet[i], dates, data)
            for x in range(numStocks):
                corrSimSetDays[x][i] = corrSetDay[x]
        initGuess = initialGuess(corrSimSetDays, len(corrSimSet))
        bnds = boundsCreator()
        con1 = {'type': 'eq' , 'fun': constraintSumOne}
        cons = [con1]
        sol = minimize(objective, initGuess, args=(corrSimSetDays, len(corrSimSet)), method='SLSQP', bounds = bnds, constraints=cons)
        if sol.success == True:
            return sol.x
        else:
            print("could not optimise so will return CORN PORT")
            tempRelative = 0
            bestDay = -1
            port = np.zeros((numStocks))
            print("I found this many in my corrSimSet: " + str(len(corrSimSet)))
            for i in corrSimSet:
                # get the price relative vector for the day
                priceRelative = dayReturn(i,dates,data)
                temp = priceRelative.max()
                if day == -1:
                    print("Error occurred at day " + str(i) + " Stuff went terribly inside expert learn")
                else:
                    if tempRelative < temp:
                        bestDay = i
                        temp = priceRelative.max()
                        port = np.zeros((numStocks))
                        # print(tempRelative)
                        port[np.argmax(priceRelative,axis=0)] = 1
            # print("WAS ABLE TO FIND AN OPTIMAL PORT")
            return port
        
def dayReturn(day, dates, data):
    """
    Given a day, the dates and a dataframe.
    Get stock market data for the given day - organise it vertically.
    TODO CHECK THAT THIS WORKS
    NOTE data here is the newly created price relative matrix for market history
    """
    day = int(day)
    if day != 0:
            # yesterdayReturn = data[data['Date'] == dates[day-1]]
            # yesterdayReturn = yesterdayReturn.Close.to_numpy()
            # todayReturn = data[data['Date'] == dates[day]]
            # todayReturn = todayReturn.Close.to_numpy()
            # todayReturn = todayReturn / yesterdayReturn
            # return todayReturn.reshape(len(todayReturn),1)
            # want a column for day before so
            # since already encoded in this format
            # print(data.shape)
            todayReturn = np.zeros((numStocks))
            # print(todayReturn.shape)
            # print(data.shape)
            try:
                for x in range(numStocks):
                    # print("X IS : " + str(x))
                    # print("TODAY RETURN AT " + str(todayReturn[x]))
                    todayReturn[x] = data[x][day]
                    # print("TODAY RETURN AT " + str(todayReturn[x]))
                # for x in range(numStocks):
                #     if math.isnan(todayReturn[x]):
                #         print("error occurred here")
                #         return np.ones((numStocks))
                return todayReturn
            except:
                print(data.shape)
                print(day)
                print(data[:][day])
                input()
    else:
        # Find number of stocks and then return 1 for each
        # startDate = data[data['Date'] == dates[0]]
        # tickers = np.unique(startDate.Ticker.to_numpy())
        numOfStocks =  data.shape[0]
        return np.ones((numOfStocks))


def getDatesVec(data):
    """
    Get the vector of dates that spans a given stock market data set - specifically done for CORN algorithm but not exclusive to it
    note that this requires the pandas dataframe of data
    NOTE pandas dataframe for data
    """
    startDate = data.Date.min()
    startDate = data[data['Date'] == startDate]
    startDate = startDate.Ticker.to_numpy()
    tick = np.unique(startDate)[0]
    tickerDates = data[data['Ticker'] == tick]
    tickerDates = np.unique(data.Date.to_numpy())
    return tickerDates

def cornDataRead():
    name = input("Name of data set\n")
    if name == "BIS":
        return np.loadtxt("./Data Sets/PriceRelatives/BISPRICERELATIVES.txt")
    elif name == "BOV":
        return np.loadtxt("./Data Sets/PriceRelatives/BOVPRICERELATIVES.txt")
    elif name == "EUR":
        return np.loadtxt("./Data Sets/PriceRelatives/EURPRICERELATIVES.txt")
    elif name == "JSE":
        return np.loadtxt("./Data Sets/PriceRelatives/JSEPRICERELATIVES.txt")
    elif name == "NAS":
        return np.loadtxt("./Data Sets/PriceRelatives/NASPRICERELATIVES.txt")
    elif name == "SP5":
        return np.loadtxt("./Data Sets/PriceRelatives/SP5PRICERELATIVES.txt")
    else:
        print("ERROR INPUT CORRECT NAME")
        return cornDataRead()

# does not rely on using pandas dataframe
def generateHistoricalMarket(data, dates, numStocks):
    """
    Function to generate a set of historical price relative vectors.
    Given a data set, the dates as a numpy array and the number of stocks in the data set.
    """
    print(len(dates))
    relatives = np.empty((numStocks, len(dates)))
    initalDay = np.ones((numStocks))
    relatives[:,0] = initalDay
    numErrors = 0
    errorDays = np.array(())
    for i in range(1,len(dates)):
        try:
            marketToday = data[data['Date'] == dates[i]]
            marketYesterday = data[data['Date'] == dates[i-1]]
            change = marketToday.Close.to_numpy()/marketYesterday.Close.to_numpy()
            change = change.reshape(numStocks)
            relatives[:,i] = change
            if i % 100 == 0:
                percent = i/len(dates)
                statement = "Percentage: " + str(percent*100) + "%, number of errors: " + str(numErrors)
                print(statement)
        except:
            numErrors += 1
            errorDays = np.append(errorDays, i)
            #acknowledge where errors occured we appeneded a 1's array
            relatives[:,i] = np.ones((numStocks))
    for i in errorDays:
        print("Error at day: " +str(i))
    print(numErrors)
    name = "SP5PRICERELATIVES.txt"
    print("Saving data as " + name)
    np.savetxt(name,relatives)
    print("Saved")
    content = np.loadtxt(name)
    print("Length of saved item was as follows(numStocks,length):" + str(relatives.shape))
    print("Loaded")
    print("Shape " + str(content.shape))
    return relatives

# relies on using day return a few times
def marketWindow(startDate, endDate, dates, data):
    """
    Return a market window from t-w to t-1 (inclusive of endpoints) therefore w in width.
    startDate is the index to start on.
    endDate is the index to end on.
    dates contains a vector of dates in the data.
    data is a capture of the data set.
    """
    # Finding out the length of the stocks is useful here
    width = endDate - startDate + 1
    # Make a window that can contain all stocks
    if(width != 1):
        market = np.empty((numStocks,width))
        count = 0
        for i in range(startDate, endDate + 1):
            window = dayReturn(i,dates,data)
            # print(window)
            for j in range(numStocks):
                market[j][count] = window[j]
            count += 1
        return market
    else:
        return dayReturn(endDate,dates,data)

# only relies on dayReturn for the use of data
def calcReturns(portfolios, dates, data, initialCapital = 1):
    """
    Function which generates returns given an initial portfolio.
    Portfolios need to be a matrix that is the width of the number of tradingdays(individual portfolios), length of the number of stocks - which describe how the portfolio looks.
    Each portfolio must be simplex, so they each are a value greater than or equal to zero, and their values sum to 1.
    """
    returns = np.array(())
    for i in range(len(dates)):
        day = dayReturn(i, dates, data)
        val = 0
        for j in range(numStocks):
            val += portfolios[i][j] * day[j]
        returns = np.append(returns, val)
    return initialCapital * returns

def initExperts(windowSize, numStocks, P):
    """
    Initialise all the experts. 
    Given a windowSize(max) assign some to each.
    For a number of stocks that are given by the tickers.
    For P where we will figure out our correlation.
    """

    # init W*P experts
    experts = []
    for i in range(0,windowSize-1):
        for j in range(P):
            # __init__(self, windowSize, corrThresh, numStocks, numDays):
            expert = Expert(i+1,j/P, numStocks, len(dates))
            experts.append(expert)
    return experts

def printExperts(experts, windowSize, P):
    """
    Function to print the experts.
    Pay attention to the indexing, since a 0 window does not make sense it feels
    """
    for i in range(0,windowSize-1):
        for j in range(0,P):
            print("Expert at " + str(i*(windowSize-1) + j) +" with characteristics:"+str(experts[i*(windowSize-1) +j].windowSize) + "," + str(experts[i*(windowSize-1) +j].corrThresh))

def findTopK(experts):
    """
    Function to find the top-K experts.
    Based on a chosen K
    An array of the indices of where the best elements occurred) NOTE THAT THIS WILL BE A FLATTENED ARRAY
    """
    expertsWealth = np.empty((windowSize-1)*P)
    for i in range((windowSize-1)*P):
        expertsWealth[i] = experts[i].wealthAchieved
            # print(experts[i*(windowSize-1) + j].wealthAchieved)
            # print(expertsWealth)
    indicesBest = np.array(())

    for i in range(K):
        currBest = np.argmax(expertsWealth)
        indicesBest = np.append(indicesBest, currBest)
        # Create a sentinel value to ignore
        expertsWealth[currBest] = -999
    return indicesBest

def beginUniformStart(dates, data, trainSize, experts, windowSize, P):
    returns = np.array(())
    returns = np.append(returns, 1)
    for i in range(0,trainSize):
        today = dayReturn(i, dates, data)
        val = today @ uniformPort
        returns = np.append(returns, val)
        print("I is: " + str(i))
        print("TOTAL RETURN AT CURRENT IS: " + str(val))

    for i in range((windowSize - 1)*P):
        experts[i].wealthAchieved = returns[-1]
    return returns, experts
# No reliance on dataframe data

def reAdjustKMeans(dataPointsWindows, data, centroidsWindows, trainSize, windowSize, P, numCluster, endDate, tol):
    #reAdjustKMeans(dataPointsWindows, data, centroidsWindows, 2*trainSize, windowSize, P, numCluster, i, tol, endDate)
    dataPointsWindows = []
    centroidsWindows = []
    print("\t READJUSTING - THE END DATE IS: " + str(endDate))
    print("\t READJUSTING - THE START DATE IS: " + str(endDate - trainSize - 1))
    print("\t READJUSTING - TRAINSIZE IS: " + str(trainSize))
    # input()
    if trainSize > 0:
        for i in range(1, windowSize+1):
            tempData = createPoints(trainSize, i, data, startDate, startDate + trainSize)
            tempCentroids = generateCentroids(numCluster)
            randomAssignCentroids(tempData, numCluster)
            reAdjustDataAssign(tempCentroids, tempData, numCluster, tol)
            dataPointsWindows.append(tempData)
            centroidsWindows.append(tempCentroids)

    return dataPointsWindows, centroidsWindows

def runCorn(dates, data, windowSize, P, trainSizeSmall, trainSizeMedium, trainSizeLarge, numClusterSmall, numClusterMedium, numClusterLarge, startDateSmall, startDateMedium, startDateLarge, totalStartDate):
    """
    Run the CORN-K algorithm on the data set
    TODO CHANGE THIS TO WORK WITH THE NEW EXPERT ARRAY AND HOW IT IS A FLAT ARRAY
    """
    # create experts which a 1D array
    expertsSmall = initExperts(windowSize,numStocks,P)
    expertsMedium = initExperts(windowSize,numStocks,P)
    expertsLarge = initExperts(windowSize,numStocks,P)
    # going downwards window size increases, going rightwards the corrThresh increases
    totReturn = 1
    # starting from first day to the final day
    # first day we get an initial wealth of 0 (t = 0)
    returns = np.array(())
    returns = np.append(returns,1)

    dataPointsWindowsSmall = []
    dataPointsWindowsMedium = []
    dataPointsWindowsLarge = []
    centroidsWindowsSmall = []
    centroidsWindowsMedium = []
    centroidsWindowsLarge = []
    
    freqRandomSmall = trainSizeSmall // 3
    freqRandomMedium = trainSizeMedium // 3
    freqRandomLarge = trainSizeLarge // 3

    # set the small data set to go
    if trainSizeSmall > 0:
        marketData = data[:,startDateSmall:startDateSmall + trainSizeSmall-1]
        for i in range(1, windowSize+1):
            # tempData = def createPoints(trainSize, kWindowSize, marketData, startDate, endDate)
            tempData = createPoints(trainSizeSmall, i, marketData, startDateSmall, startDateSmall + trainSizeSmall)
            print("==============FOR INITIALISATION==============")
            print("\t Start Date is: " + str(startDateSmall))
            print("\t End Date is: " + str(startDateSmall + trainSizeSmall - 1))
            print("\t TrainSize is: " + str(trainSizeSmall - 1))
            # input()
            tempCentroids = generateCentroids(numClusterSmall)
            randomAssignCentroids(tempData, numClusterSmall)
            reAdjustDataAssign(tempCentroids, tempData, numClusterSmall, tol)
            dataPointsWindowsSmall.append(tempData)
            centroidsWindowsSmall.append(tempCentroids)
    if numClusterSmall == 0:
        numClusterSmall = 1

    # set the medium data set to go
    if trainSizeMedium > 0:
        marketData = data[:,startDateMedium:startDateMedium + trainSizeMedium-1]
        for i in range(1, windowSize+1):
            # tempData = def createPoints(trainSize, kWindowSize, marketData, startDate, endDate)
            tempData = createPoints(trainSizeMedium, i, marketData, startDateMedium, startDateMedium + trainSizeMedium)
            print("==============FOR INITIALISATION==============")
            print("\t Start Date is: " + str(startDateMedium))
            print("\t End Date is: " + str(startDateMedium + trainSizeMedium - 1))
            print("\t TrainSize is: " + str(trainSizeMedium - 1))
            # input()
            tempCentroids = generateCentroids(numClusterMedium)
            randomAssignCentroids(tempData, numClusterMedium)
            reAdjustDataAssign(tempCentroids, tempData, numClusterMedium, tol)
            dataPointsWindowsMedium.append(tempData)
            centroidsWindowsMedium.append(tempCentroids)
    if numClusterMedium == 0:
        numClusterMedium = 1
    # set the large data set to go
    if trainSizeLarge > 0:
        marketData = data[:,startDateLarge:startDateLarge + trainSizeLarge-1]
        for i in range(1, windowSize+1):
            # tempData = def createPoints(trainSize, kWindowSize, marketData, startDate, endDate)
            tempData = createPoints(trainSizeLarge, i, marketData, startDateLarge, startDateLarge + trainSizeLarge)
            print("==============FOR INITIALISATION==============")
            print("\t Start Date is: " + str(startDateLarge))
            print("\t End Date is: " + str(startDateLarge + trainSizeLarge - 1))
            print("\t TrainSize is: " + str(trainSizeLarge - 1))
            # input()
            tempCentroids = generateCentroids(numClusterLarge)
            randomAssignCentroids(tempData, numClusterLarge)
            reAdjustDataAssign(tempCentroids, tempData, numClusterLarge, tol)
            dataPointsWindowsLarge.append(tempData)
            centroidsWindowsLarge.append(tempCentroids)
    if numClusterLarge == 0:
        numClusterLarge = 1
    # returns, experts = beginUniformStart(dates, data, trainSize, experts, windowSize, P)

    for i in range(totalStartDate,len(dates)):
        print("I is: " + str(i))
        # for each window size as based on the experts which is of length windowSize - 1
        for w in range((windowSize - 1)*P):
            tempPointsSmall = dataPointsWindowsSmall[w//P]
            tempPointsMedium = dataPointsWindowsMedium[w//P]
            tempPointsLarge = dataPointsWindowsLarge[w//P]
            expertsSmall[w].currPort = expertLearn(expertsSmall[w].windowSize, expertsSmall[w].corrThresh, i, data, tempPointsSmall, "Small")
            expertsMedium[w].currPort = expertLearn(expertsMedium[w].windowSize, expertsMedium[w].corrThresh, i, data, tempPointsMedium, "Medium")
            expertsLarge[w].currPort = expertLearn(expertsLarge[w].windowSize, expertsLarge[w].corrThresh, i, data, tempPointsLarge, "Large")
        # combine our experts' portfolios
        for w in range(1,windowSize+1):
            i
            marketData = data[:,i-w-1:i-1]
            # add the new data to all of them
            addNewDataPoint(centroidsWindowsSmall[w-1], dataPointsWindowsSmall[w-1], marketData, i, w)
            addNewDataPoint(centroidsWindowsMedium[w-1], dataPointsWindowsMedium[w-1], marketData, i, w)
            addNewDataPoint(centroidsWindowsLarge[w-1], dataPointsWindowsLarge[w-1], marketData, i, w)
            # deal with small window
            if i % freqRandomSmall == 0:
                centroids = generateCentroids(numClusterSmall)
                centroidsWindowsSmall[w-1] = centroids
                randomAssignCentroids(dataPointsWindowsSmall[w-1], numClusterSmall)
                reAdjustDataAssign(centroidsWindowsSmall[w-1], dataPointsWindowsSmall[w-1], numClusterSmall, tol)
            # deal with medium window
            if i % freqRandomMedium == 0:
                centroids = generateCentroids(numClusterMedium)
                centroidsWindowsMedium[w-1] = centroids
                randomAssignCentroids(dataPointsWindowsMedium[w-1], numClusterMedium)
                reAdjustDataAssign(centroidsWindowsMedium[w-1], dataPointsWindowsMedium[w-1], numClusterMedium, tol)
            # deal with large window
            if i % freqRandomLarge == 0:
                centroids = generateCentroids(numClusterLarge)
                centroidsWindowsLarge[w-1] = centroids
                randomAssignCentroids(dataPointsWindowsLarge[w-1], numClusterLarge)
                reAdjustDataAssign(centroidsWindowsLarge[w-1], dataPointsWindowsLarge[w-1], numClusterLarge, tol)

        #  deal with small window
        if i % 3 * freqRandomSmall == 0:
            numClusterSmall += 1
        # deal with medium window
        if i % 3 * freqRandomMedium == 0:
            numClusterMedium += 1
        # deal with large window
        if i % 3 * freqRandomLarge == 0:
            numClusterLarge += 1

        # small size
        if i % (2*trainSizeSmall) == 0:
            numClusterSmall = 4
            marketWindow = data[:,i-trainSizeSmall-1:i-1]
            dataPointsWindowsSmall, centroidsWindowsSmall = reAdjustKMeans(dataPointsWindowsSmall, marketWindow, centroidsWindowsSmall, trainSizeSmall, windowSize, P, numClusterSmall, i, tol)
            print("========READJUSTED SMALL-SIZE DATASET========")
        # medium size
        if i % (2*trainSizeMedium) == 0:
            numClusterMedium = 4
            marketWindow = data[:,i-trainSizeMedium-1:i-1]
            dataPointsWindowsMedium, centroidsWindows = reAdjustKMeans(dataPointsWindowsMedium, marketWindow, centroidsWindowsMedium, trainSizeMedium, windowSize, P, numClusterMedium, i, tol)
            print("========READJUSTED MEDIUM-SIZE DATASET========")
        # large size
        if i % (2*trainSizeLarge) == 0:
            numClusterLarge = 4
            marketWindow = data[:,i-trainSizeLarge-1:i-1]
            dataPointsWindowsLarge, centroidsWindowsLarge = reAdjustKMeans(dataPointsWindowsLarge, marketWindow, centroidsWindowsLarge, trainSizeLarge, windowSize, P, numClusterLarge, i, tol)
            print("========READJUSTED LARGE-SIZE DATASET========")

        day = dayReturn(i, dates, data)
        #update the experts' individual wealths

        # update each set of the experts from small to the large
        
        expertDayEarlySmall = expertsSmall
        expertDayEarlyMedium = expertsMedium
        expertDayEarlyLarge = expertsLarge
        # small
        for m in range((windowSize-1)*P):
            expertsSmall[m].increaseWealth(day)
            expertsMedium[m].increaseWealth(day)
            expertsLarge[m].increaseWealth(day)

        # TOP-K and expert weights update
        # first need to find these top-K experts
        # so select top K experts based on historical performance - so search through experts and find their wealths, as a 2D matrix, find those indices and work backwards ?
        # this will not be a 2D array and instead an array that is flattened
        # Given that experts should also be a flattened array this should be acceptable
        # do this for each set of experts

        #small, medium, large
        topKSmall = findTopK(expertDayEarlySmall)
        topKMedium = findTopK(expertDayEarlyMedium)
        topKLarge = findTopK(expertDayEarlyLarge)

        # since topK contains the indices of the top-k experts we will just loop through the experts
        for x in topKSmall:
            # set their weights (TOP K)
            x = int(x)
            if x in topKSmall:
                expertDayEarlySmall[x].weight = 1 / K
            # just not setting the weights for the others should acheive the same complexity
        for x in topKMedium:
            # set their weights (TOP K)
            x = int(x)
            if x in topKMedium:
                expertDayEarlyMedium[x].weight = 1 / K
            # just not setting the weights for the others should acheive the same complexity
        for x in topKLarge:
            # set their weights (TOP K)
            x = int(x)
            if x in topKLarge:
                expertDayEarlyLarge[x].weight = 1 / K
            # just not setting the weights for the others should acheive the same complexity

        todayPortNumeratorSmall = np.zeros(numStocks)
        todayPortDenomSmall = np.zeros(numStocks)
        for x in topKSmall:
            x = int(x)
            if expertDayEarlySmall[x].weight != 0:
                todayPortNumeratorSmall += expertDayEarlySmall[x].weight * (expertDayEarlySmall[x].wealthAchieved * expertDayEarlySmall[x].currPort)
                todayPortDenomSmall += expertDayEarlySmall[x].weight * expertDayEarlySmall[x].wealthAchieved
            else:
                pass
        todayPortSmall = todayPortNumeratorSmall / todayPortDenomSmall

        todayPortNumeratorMedium = np.zeros(numStocks)
        todayPortDenomMedium = np.zeros(numStocks)
        for x in topKMedium:
            x = int(x)
            if expertDayEarlyMedium[x].weight != 0:
                todayPortNumeratorMedium += expertDayEarlyMedium[x].weight * (expertDayEarlyMedium[x].wealthAchieved * expertDayEarlyMedium[x].currPort)
                todayPortDenomMedium += expertDayEarlyMedium[x].weight * expertDayEarlyMedium[x].wealthAchieved
            else:
                pass
        todayPortMedium = todayPortNumeratorMedium / todayPortDenomMedium 

        todayPortNumeratorLarge = np.zeros(numStocks)
        todayPortDenomLarge = np.zeros(numStocks)
        for x in topKLarge:
            x = int(x)
            if expertDayEarlyLarge[x].weight != 0:
                todayPortNumeratorLarge += expertDayEarlyLarge[x].weight * (expertDayEarlyLarge[x].wealthAchieved * expertDayEarlyLarge[x].currPort)
                todayPortDenomLarge += expertDayEarlyLarge[x].weight * expertDayEarlyLarge[x].wealthAchieved
            else:
                pass
        todayPortLarge = todayPortNumeratorLarge / todayPortDenomLarge 

        val = day @ ( (todayPortSmall + todayPortMedium + todayPortLarge) / 3 )
        if not math.isnan(val):
            totReturn = totReturn * val
        else:
            print("NAN VALUE ENCOUNTERED AT DATE:" + str(i))
        print("TOTAL RETURN AT CURRENT IS: " + str(totReturn))
        returns = np.append(returns,totReturn)

        # if val == 0:
        #     print("VALUE IS 0 AT DAY" + str(i))
        if i == ENDdate:
            return returns
    return returns

# find the best amongst the best
def findBestAmongstTopK(topKSmall, smallExperts, topKMedium, mediumExperts, topKLarge, largeExperts):
    
    
    
    bestExpertArray = np.empty((3,5))
    count = 0
    for k in range(len(topKSmall)):
        bestExpertArray[0,count] = smallExperts[int(topKSmall[k])].wealthAchieved
        bestExpertArray[1,count] = mediumExperts[int(topKMedium[k])].wealthAchieved
        bestExpertArray[2,count] = largeExperts[int(topKLarge[k])].wealthAchieved
        count += 1
    
    bestExpertArray = bestExpertArray.flatten()
    bestIndices = np.array(())
    for k in range(K):
        currBest = np.argmax(bestExpertArray)
        bestIndices = np.append(bestIndices, currBest)
        bestExpertArray[currBest] = -999

    bestAgents = []
    for i in bestIndices:
        i = int(i)
        if i < 5:
            bestAgents.append(smallExperts[i])
        elif i < 10:
            bestAgents.append(mediumExperts[i-5])
        elif i <  15:
            bestAgents.append(largeExperts[i-10])
    return bestAgents       

# new stuff from our kMeansMarket to test this out
def createPoints(trainSize, kWindowSize, marketData, startDate, endDate):
    """
    A function that creates data points using a given window size and a market of price relatives.
    """
    dataPoints = []
    print("End date is: " + str(trainSize))
    for i in range(kWindowSize,endDate-startDate):
        data = marketData[:,i-kWindowSize:i]
        point = kDataPoint(data, kWindowSize, i)
        dataPoints.append(point)
    return dataPoints

def reAdjustTraining(endDate, trainSize, kWindowSize, marketData):
    """
    A function that creates data points using a given window size and a market of price relatives.
    """
    dataPoints = []
    for i in range(trainSize, endDate):
        data = marketData[:,i-kWindowSize:i]
        point = kDataPoint(data, kWindowSize, i)
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
    # print(before)
    for i in dataPoints:
        i.reassignCluster(centroids)
    after = currentAssignments(dataPoints)
    # print(after)
    diff = after - before
    noChange = len(np.where(diff != 0)[0])
    change = noChange / len(dataPoints)
    # print("Percentage changed: " + str(change*100) + "%")
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
    numAdjusts = 0
    while currError > tol:
        centroids = updateCentroids(centroids, dataPoints, numCluster)
        dataPoints, currError = updateClusters(centroids, dataPoints)
        numAdjusts += 1
        if numAdjusts > 10:
            return

def addNewDataPoint(centroids, dataPoints, marketWindow, day, kWindowSize):
    point = kDataPoint(marketWindow, kWindowSize, day)
    point.reassignCluster(centroids)
    dataPoints.append(point)
    return dataPoints

def visualiseDataBins(start, end):
    "Visualise the change in datapoints from start to end. Pass them as they looked at the start and what they looked like at the end"
    plt.subplot(1, 2, 1)
    plt.hist(start, bins=numCluster-1)
    plt.title("Cluster Assignment Start")
    plt.ylabel("Num Data Points")
    plt.xlabel("Cluster ID")
    plt.subplot(1,2,2)
    plt.hist(end, bins=numCluster-1)
    plt.title("Cluster Assignment End")
    plt.ylabel("Num Data Points")
    plt.xlabel("Cluster ID")
    plt.show()

def visualiseDataBinsThree(start, second, end):
    "Visualise the change in datapoints from start to end. Pass them as they looked at the start and what they looked like at the end"
    plt.subplot(1, 3, 1)
    plt.hist(start, bins=numCluster-1)
    plt.title("Cluster Assignment Start")
    plt.ylabel("Num Data Points")
    plt.xlabel("Cluster ID")
    plt.subplot(1, 3, 2)
    plt.hist(second, bins=numCluster-1)
    plt.title("Cluster Assignment Middle")
    plt.ylabel("Num Data Points")
    plt.xlabel("Cluster ID")
    plt.subplot(1,3,3)
    plt.hist(end, bins=numCluster-1)
    plt.title("Cluster Assignment End")
    plt.ylabel("Num Data Points")
    plt.xlabel("Cluster ID")
    plt.show()

def returnSimilarDays(day, dataPoints):
    """
    Get the days from our datapoints that are part of the same cluster day as today.
    """
    similarDays = np.array(())
    cluster = dataPoints[day].kCluster
    for i in dataPoints:
        if cluster == i.kCluster:
            similarDays = np.append(similarDays, i.day)
    print(similarDays)
    return similarDays

data = readDataSet()
dataset = cornDataRead()
for i in range(dataset.shape[1]):
    for j in range(dataset.shape[0]):
        if math.isnan(dataset[j][i]):
            dataset[j][i] = 1

dates = getDatesVec(data)
tempStartFind = data[data['Date'] == dates[0]]
tempTickersFind = np.unique(tempStartFind.Ticker.to_numpy())
numStocks = len(tempTickersFind)
market = marketWindow(1,1,dates,dataset)
uniformPort = np.ones((numStocks)) / numStocks
testDateEnd = 708
market = input("Input the market \n")
# make sure to make it possible to interact with validation data correctly


windowSize = 5
P = 10
K = 5
tol = 1e-2
trainSizeSmall = 10
trainSizeMedium = 120
trainSizeLarge = 190



numClusterSmall = trainSizeSmall // 3
numClusterMedium = trainSizeMedium // 3
numClusterLarge = trainSizeLarge // 3

startDateCount = 2
averageTradeDay = 254
startDate = 200
startDateSmall = startDate - trainSizeSmall
startDateMedium = startDate - trainSizeMedium
startDateLarge = startDate - trainSizeLarge
ENDdate = startDate + (2 * averageTradeDay)
count = 0

wealth = runCorn(dates, dataset, windowSize, P, trainSizeSmall, trainSizeMedium, trainSizeLarge, numClusterSmall, numClusterMedium, numClusterLarge, startDateSmall, startDateMedium, startDateLarge, startDate)


name = "mixed-model"
np.savetxt("./Data Sets/TrainVal-start-{0}-end-{1}-{2}-sizes{3}-{4}-{5}.txt".format(startDate, ENDdate, market, name, trainSizeSmall, trainSizeMedium, trainSizeLarge),wealth)