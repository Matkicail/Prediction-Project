from datetime import date
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stockMarketReader import readDataSet

def initPriceRelativeVec(data, t, dates):
    """
    Given a data set at a specific date as indicated by t, which is the index position of the date.
    It will use the dates vector - of all the dates that have occurred to get a market relative vector
    """
    marketToday = data[data['Date'] == dates[t]]
    marketToday = marketToday.Close.to_numpy()
    marketYesterday = data[data['Date'] == dates[t-1]]
    marketYesterday = marketYesterday.Close.to_numpy()
    return marketToday/marketYesterday

def initMarketMovements(data, window, t, dates, numStocks):
    """
    Get a set of market relative vectors - called the market window for the ith trading day.
    Note that the priceRelatives required to do this are t-w, t-w+1, ... , t-2 , t-1.
    """
    count = t - window
    marketWindow = np.empty((numStocks, (t-1) - (t-window)))
    i = 0
    while count < t:
        # Do stuff
        marketWindow[i,:] = initPriceRelativeVec(data, count, dates)
        # Next day
        count += 1
        i += 1

def CORNK(data):
    """
    CORN-K strategy as based on by the CORN-K paper Li et al.
    """
    # Basic set up of general variables: dates, starting prices, uniform portfolio, number of stocks
    startDate = data.Date.min()
    startPrices = data[data['Date'] == startDate]
    numStocks = len(np.unique(startPrices.Ticker.to_numpy()))
    tick = np.unique(startPrices.Ticker.to_numpy())[0]
    dates = data[data['Ticker'] == tick]
    dates = dates.Date.to_numpy()
    propPort = 1 / numStocks
    # Prop owned start acts like the uniform portfolio - in the original capital is divided equally amongst all assets -> 1/numStock
    # But obviously different percentages of each asset can be bought for this amount, therefore the propOwned will have different percentages
    propOwnedStart = propPort / startPrices.Close.to_numpy()

def cornDay(data, day, dates, window, coeffThresh, currPort):
    """
    Preform a single day for the CORN-K algorithm. Should be noted that the following inputs are required:
    data - the stock market data for time period required.
    day - the index date we are at.
    dates - vector of dates as a numpy array
    window - window size (number of days)
    coeffThresh - a hyper parameter which is based on the chosen coefficient threshold.
    currPort - the portfolio from the previous trading day which must be updated
    """
    # Not enough data to do a corn day, so return uniform portfolio
    if day <= window + 1:
        return currPort
    for i in range(window + 1, day):

        # if the correlation coeff, of the ___ with ___ is greater than the threshold then
        pass

    # if the 

data = readDataSet()



