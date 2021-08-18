from datetime import date
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stockMarketReader import readDataSet

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

def bestStockStrategy(data, startAt, stopAt):
    """
    Given a specific stock exchange, this will find the best stock, record the returns of the best stock daily and display it as a plot.
    The returns are however available as they will be returned from this function.
    """

    startDate = data.Date.min()
    
    dates = getDatesVec(data)
    startDate = dates[startAt]
    endDate = dates[stopAt]
    
    data = data[data["Date" ] >= startDate]
    # print(data.head(10))
    data = data[data["Date" ] <= endDate]
    # print(data.head(10))
    startPrices = data[data['Date'] == startDate]
    startOfMarket = startPrices.copy()
    startPrices = startPrices.Close.to_numpy()
    endPrices = data[data['Date'] == endDate]
    endPrices = endPrices.Close.to_numpy()
    relativeChange = endPrices / startPrices
    indexMax = np.where(relativeChange == relativeChange.max())[0][0]
    tickers = startOfMarket.Ticker.to_numpy()
    initProp = 1 / startOfMarket.Close.to_numpy()[indexMax]
    # Need to grab the right stock at the start date, so two parts to do here
    returns = np.array(())
    # Here we are grabbing the data of the ticker that achieved maximal returns
    
    marketData = data[data['Ticker'] == tickers[indexMax]]
    # marketData = marketData[data['Date'] >= startDate]
    # marketData = marketData[data['Date'] < endDate]
    print("HERE")
    print(marketData.head(10))
    dates = marketData.Date.to_numpy()
    print(dates)
    for i in dates:
        currDay = marketData[marketData['Date'] == i]
        returns = np.append(returns, initProp * currDay.Close)
    # plt.ylabel("Multiple of Increase")
    # plt.xlabel("Number of Days Passed")
    # plt.title("Total Return of " + tickers[indexMax])    
    # plt.plot(returns)
    # plt.show()
    return returns, tickers[indexMax]

# data = readDataSet()
# bestStockStrategy(data)