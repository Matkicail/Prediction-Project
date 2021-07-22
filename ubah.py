from datetime import date
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stockMarketReader import readDataSet

def ubah(data):
    """
    Given a data set, preform a uniform buy and hold strategy.
    Thus divide the initial capital (1) amongst all assets equally and track the return.
    This will return the total return at each day, sequentially for a universal portfolio.
    """
    startDate = data.Date.min()
    startPrices = data[data['Date'] == startDate]
    numStocks = len(np.unique(startPrices.Ticker.to_numpy()))
    tick = np.unique(startPrices.Ticker.to_numpy())[0]
    print("Ticker ")
    print(tick)
    dates = data[data['Ticker'] == tick]
    print(dates)
    dates = dates.Date.to_numpy()
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
    plt.ylabel("Multiple of Increase")
    plt.xlabel("Number of Days Passed")
    plt.plot(returns)
    plt.show()
    return returns

data = readDataSet()
vals = ubah(data)