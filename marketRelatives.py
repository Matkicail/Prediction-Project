from datetime import date
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def marketRelativeVectors(data):
    """
    Create a market relative vector which records daily price changes based on the previous day to today's change - i.e X_t / X_t-1.
    It will return this so that you can construct a daily return with a portfolio that continuously changes.
    It will need to be informed what data set it ran on in cases where you wish to write it or do something extra - as it will not record names or that sort.
    """
    startDate = data.Date.min()
    startPrices = data[data['Date'] == startDate]
    tick = np.unique(startPrices.Ticker.to_numpy())[0]
    numStocks = len(np.unique(startPrices.Ticker.to_numpy()))
    dates = data[data['Ticker'] == tick].to_numpy()
    initialChange = np.ones((numStocks,1))
    count = 0
    relativeHistory = np.array(())
    relativeHistory = np.append(relativeHistory, initialChange)
    numError = 0
    
    for i in dates:
        if count == 0:
            count += 1
        else:
            currDay = data[data['Date'] == i[0]]
            currDay = currDay.Close.to_numpy()
            # print(dates[count-1])
            prevDay = data[data['Date'] == dates[count - 1][0]]
            prevDay = prevDay.Close.to_numpy()
            count += 1
            relativeChange = currDay / prevDay
            
            try:
                relativeHistory = np.hstack((relativeHistory, relativeChange))
            except:
                numError += 1
    if numError != 0:
        print("Number of errors " + str(numError))
    else:
        print("No Errors occured in recording historical relative prices")
    return relativeHistory