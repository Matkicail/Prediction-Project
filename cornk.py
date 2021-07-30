from datetime import date
import numpy as np
from numpy.lib import corrcoef
import pandas as pd
from matplotlib import pyplot as plt
from stockMarketReader import readDataSet

def getUniformPort():
    stocks = np.ones((numStocks))
    return stocks / numStocks

def expertLearn(window, corrThresh, histMarketWind, day):
    """
    Preform algorithm 1 from CORN paper.
    This algorithm is the expert learning procedure.
    Given an index date (day), the window (a specified window size), the histMarketWind (t-1 to 1) and the corrThresh which is rho our correlation coeff threshold.
    """
    corrSimSet = set()
    if day <= window + 1:
        return getUniformPort
    else:
        for i in range(window + 1,day): #just check that this works otherwise change it to t
            markWindI = marketWindow(i-window, i-1, dates, data)
            markWindT = marketWindow(day - window, day - 1, dates, data)
            if np.corrcoef(markWindI, marketWindow) >= corrThresh:
                # append this to our set 
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
    Return a market window from t-w to t-1 (inclusive of endpoints).
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

data = readDataSet()
dates = getDatesVec(data)
tempStartFind = data[data['Date'] == dates[0]]
tempTickersFind = np.unique(tempStartFind.Ticker.to_numpy())
numStocks = len(tempTickersFind)
today = dayReturn(1,dates,data)
market = marketWindow(1007,1012,dates,data, numStocks)
print(market)

