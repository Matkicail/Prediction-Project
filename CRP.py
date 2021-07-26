from datetime import date
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stockMarketReader import readDataSet
# ######################################## #
# Implementing the CRP Portfolio Benchmark #
# ######################################## #

def UCRP(data):
    """
    A function that given a data set will return a constantly rebalanced portfolio.
    TODO check the JSE returns to make sure this program actually works, feels a bit dodgy on that one for that day.
    NOTE THIS IS THE UNIFORM CONSTANTLY REBALANCED PORTFOLIO - AS DESCRIBED IN CORN PAPER
    """
    startDate = data.Date.min()
    startPrices = data[data['Date'] == startDate]
    numStocks = len(np.unique(startPrices.Ticker.to_numpy()))
    tick = np.unique(startPrices.Ticker.to_numpy())[0]
    dates = data[data['Ticker'] == tick]
    dates = dates.Date.to_numpy()
    propPort = 1 / numStocks
    currPort = propPort / startPrices.Close.to_numpy()
    returns = np.array(())
    numError = 0
    for i in range(len(dates)):
        if i == 0:
            returns = np.append(returns, 1)
        else:
            try:
                currMarket = data[data['Date'] == dates[i]]
                currMarket = currMarket.Close.to_numpy()
                dayReturn = np.dot(currPort.T,currMarket)
                returns = np.append(returns, dayReturn)
                propPort = dayReturn / numStocks
                currPort = propPort / currMarket
            except:
                # print("Error occured")
                numError +=1 
    # print("Total number of errors " + str(numError))
    # plt.title("CRP")
    # plt.ylabel("Multiple of Increase")
    # plt.xlabel("Number of Days Passed")
    # plt.plot(returns)
    # plt.show()
    return returns
# data = readDataSet()
# crpReturns = CRP(data)
