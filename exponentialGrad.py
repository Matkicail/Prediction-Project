# This is an implementation of Exponential Gradient (an may be wrong as it is based on my understanding)
# This is from D. P. Helmbold, R. E. Schapire, Y. Singer, and M. K. Warmuth, “On-line portfolio selection using multiplicative updates

from datetime import date
import numpy as np
from numpy.lib import corrcoef
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import data
from stockMarketReader import readDataSet
from scipy.optimize import minimize 
import math


def constraintSumOne(portfolio):
    prob = 1
    for i in portfolio:
        prob -= i
    return prob

def boundsCreator():
    b = (0.0,1.0)
    a = [b]*numStocks
    return a

def initialGuess(day, t):
    port = np.ones(numStocks) / numStocks
    return port

def dataRead():
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
        return dataRead()

def dayReturn(day, data):
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

def R(b, bCurr):
    """
    Regularising term
    """
    term = np.sum( b * np.log(b/bCurr))
    return term
def objective(portfolio, bCurr ,day, ita = 0.05):
    """
    Ensuring that days is a vector/matrix where width is number of days and length is numStocks.
    """
    mainTerm = bCurr @ day
    total = ita * (np.log(mainTerm) + (day/mainTerm) @ (portfolio - bCurr)) - R(portfolio,bCurr)

    # Return negative portfolio so that we can minimise (hence maximising the portfolio)
    return -total


def runExponential(dataSet, startDate, endDate):
    """
    Run the exponential gradient algorithm
    """
    totReturn = 1
    wealth = np.array(())
    portfolio = np.ones(dataSet.shape[0])/ dataSet.shape[0]
    for i in range(startDate, endDate):
        today = dataSet[:,i]
        initGuess = initialGuess(dataSet, i)
        bnds = boundsCreator()
        con1 = {'type': 'eq' , 'fun': constraintSumOne}
        cons = [con1]
        sol = minimize(objective, initGuess, args=(portfolio, today), method='SLSQP', bounds = bnds, constraints=cons)
        portfolio = np.ones(numStocks)
        if sol.success == True:
            portfolio = sol.x
        else:
            portfolio = np.ones(numStocks) / numStocks
            print("Had to return uniform")
        totReturn = (totReturn * portfolio) @ dataSet[:,i]
        wealth = np.append(wealth, totReturn)
        print("Return at day {0} is {1}".format(i, totReturn))
    
    return wealth





trainSize = 100
startDate = 200
endDate = 708
market = input("Which market ? \n")
dataSet = dataRead()
for i in range(dataSet.shape[1]):
    for j in range(dataSet.shape[0]):
        if math.isnan(dataSet[j][i]):
            dataSet[j][i] = 1
numStocks = dataSet.shape[0]
wealth = runExponential(dataSet, startDate, endDate)
