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

def ubah(data, startAt, stopAt):
    """
    Given a data set, preform a uniform buy and hold strategy.
    Thus divide the initial capital (1) amongst all assets equally and track the return.
    This will return the total return at each day, sequentially for a universal portfolio.
    """
    dates = getDatesVec(data)
    startDate = dates[startAt]
    endDate = dates[stopAt]
    dates = dates[startAt:stopAt]
    startPrices = data[data['Date'] == startDate]
    numStocks = len(np.unique(startPrices.Ticker.to_numpy()))
    tick = np.unique(startPrices.Ticker.to_numpy())[0]
    # print("Ticker ")
    # print(tick)
    data = data[data['Date'] >= startDate]
    data = data[data['Date'] < endDate]
    dates = data[data['Ticker'] == tick]
    print(dates)
    dates = dates.Date.to_numpy()
    # dates = dates[startDate:endDate]
    propPort = 1 / numStocks
    # getting the init prop of a stock we can own
    # this will assume that this will come out in the right order - which it should
    # Prop owned start acts like the uniform portfolio - in the original capital is divided equally amongst all assets -> 1/numStock
    # But obviously different percentages of each asset can be bought for this amount, therefore the propOwned will have different percentages
    propOwnedStart = propPort / startPrices.Close.to_numpy()
    returns = np.array(())
    for i in dates:
        currDay = data[data['Date'] == i]
        prices = currDay.Close.to_numpy()
        try:
            dayVal = np.sum(propOwnedStart * prices)
            returns = np.append(returns, dayVal)
        except:
            returns = np.append(returns, returns[-1])
    # plt.ylabel("Multiple of Increase")
    # plt.xlabel("Number of Days Passed")
    # plt.plot(returns)
    # plt.show()
    return returns

# data = readDataSet()
# vals = ubah(data)