from datetime import date
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stockMarketReader import readDataSet

def bestStockStrategy(data):
    """
    Given a specific stock exchange, this will find the best stock, record the returns of the best stock daily and display it as a plot.
    The returns are however available as they will be returned from this function.
    """
    startDate = data.Date.min()
    endDate = data.Date.max()
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
    dates = marketData.Date.to_numpy()
    for i in dates:
        currDay = marketData[marketData['Date'] == i]
        returns = np.append(returns, initProp * currDay.Close)
    plt.ylabel("Multiple of Increase")
    plt.xlabel("Number of Days Passed")
    plt.title("Total Return of " + tickers[indexMax])    
    plt.plot(returns)
    plt.show()
    return returns

data = readDataSet()
bestStockStrategy(data)