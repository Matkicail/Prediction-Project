from datetime import date
import numpy as np
from numpy.lib import corrcoef
import pandas as pd
from matplotlib import pyplot as plt
from stockMarketReader import readDataSet

# Need to construct a set of experts as required by CORN
class Expert:

    #constructor for this expert with the two noteworthy parameters
    def __init__(self, windowSize, corrThresh):
        self.windowSize = windowSize
        self.corrThresh = corrThresh
        self.corrSimSet = None
    
    def assignCorrSet(self, corrSet):
        self.corrSimSet = corrSet
    
    

def getUniformPort():
    stocks = np.ones((numStocks))
    return stocks / numStocks

def expertLearn(window, corrThresh, histMarketWind, day):
    """
    Preform algorithm 1 from CORN paper.
    This algorithm is the expert learning procedure.
    Given an index date (day), the window (a specified window size), the histMarketWind (t-1 to 1) and the corrThresh which is rho our correlation coeff threshold.
    """
    corrSimSet = np.array()
    if day <= window + 1:
        return getUniformPort
    else:
        for i in range(window + 1,day): #just check that this works otherwise change it to t
            markWindI = marketWindow(i-window, i-1, dates, data)
            markWindT = marketWindow(day - window, day - 1, dates, data)
            # check at some point to ensure that this captures the standard deviation for the whole window (i.e output not something weird)
            # flattened just to ensure that this does happen
            if np.std(np.flatten(markWindI)) == 0 or np.std(np.flatten(markWindT)) == 0:
                corrThresh = 0
            # may need to change this to the exact calculation they use in the formula
            if np.corrcoef(markWindI, marketWindow) >= corrThresh:
                # append this to our set i.e add the index
                corrSimSet = np.append(corrSimSet,i)
    if len(corrSimSet) == 0:
        return getUniformPort()
    else:
        # Search for the optimal portfolio
        # so using the i in the set, get the argmax
        # from what I understand, we need the price relative vector at time i, find the stock that gave the best return and all in on that stock
        tempRelative = 0
        port = np.zeros((numStocks))
        for i in corrSimSet:
            # get the price relative vector for the day
            priceRelative = dayReturn(i,dates,data)
            # index of maximum change 
            day = dayReturn(i,dates,data)
            day = day.reshape((numStocks,1))
            dayVal = np.argmax(day)
            if day == -1:
                print("Error occurred at day " + str(i))
            else:
                if tempRelative < day:
                    tempRelative = day
                    port = np.zeros((numStocks))
                    port[np.argmax(day,axis=0)] = 1
        return port
        
def dayReturn(day, dates, data):
    """
    Given a day, the dates and a dataframe.
    Get stock market data for the given day - organise it vertically.
    """
    if day != 0:
        try:
            yesterdayReturn = data[data['Date'] == dates[day-1]]
            yesterdayReturn = yesterdayReturn.Close.to_numpy()
            todayReturn = data[data['Date'] == dates[day]]
            todayReturn = todayReturn.Close.to_numpy()
            todayReturn = todayReturn / yesterdayReturn
            return todayReturn.reshape(len(todayReturn),1)
        except:
            print("error for given date, returning -1 as sentinel")
            return -1
    else:
        # Find number of stocks and then return 1 for each
        startDate = data[data['Date'] == dates[0]]
        tickers = np.unique(startDate.Ticker.to_numpy())
        return np.ones((len(tickers)))

def getDatesVec(data):
    """
    Get the vector of dates that spans a given stock market data set - specifically done for CORN algorithm but not exclusive to it
    """
    startDate = data.Date.min()
    startDate = data[data['Date'] == startDate]
    tick = np.unique(startDate.Ticker.to_numpy())[0]
    tickerDates = data[data['Ticker'] == tick]
    tickerDates = data.Date.to_numpy()
    return tickerDates

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
    market = np.empty((numStocks,width))
    count = 0
    for i in range(startDate, endDate + 1):
        window = dayReturn(i,dates,data)
        # print(window)
        for j in range(numStocks):
            market[j][count] = window[j]
        count += 1
    return market

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
data = readDataSet()
dates = getDatesVec(data)
tempStartFind = data[data['Date'] == dates[0]]
tempTickersFind = np.unique(tempStartFind.Ticker.to_numpy())
numStocks = len(tempTickersFind)
today = dayReturn(1,dates,data)
market = marketWindow(1007,1012,dates,data, numStocks)
print(market)

