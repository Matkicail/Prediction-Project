from datetime import date
from os import read
import numpy as np
from numpy.core.records import array
import pandas as pd
from matplotlib import pyplot as plt

def findBestStartDate(data):
    """
    Given a data set find the maximum start date, using the set of minimum dates from each ticker.
    I.e - Given a data set, find the min start date for each ticker, find the maximum of this.
    Print which stock ticker had this to check if this may be a problem so that it can be excluded.
    """
    dateInfoMin = data.groupby('Ticker').agg({'Date' : "min"})
    print("The earliest start date for this data is " + str(dateInfoMin.Date.max()))
    print(dateInfoMin.head(50))
    bestDate = dateInfoMin["Date"].to_numpy().max()
    return bestDate
    
    
def createUniformStart(data):
    """
    Given a data set, get a uniform starting date that works the best for including a large segment of the data set.
    Thus this will yield a data set with (in some cases a fair amount) fewer securities. 
    These excluded securities can be found inside readDataSet as a comment in the relevant section.
    """
    # Set the date column to be a date time variable
    data["Date"] = pd.to_datetime(data['Date'])
    bestStartDate = findBestStartDate(data)
    uniformStart = data[data["Date"] >= bestStartDate]
    return uniformStart

def enforcePositiveClose(data):
    """
    Function that will take in a cleaned data set and then make sure that all the close values are greater than zero.
    """
    dataNegative = data[data["Close"] <= 0]
    tickers = np.unique(dataNegative.Ticker.to_numpy())
    print("Checking values where negative close prices occur")
    print(tickers)
    if len(tickers) > 0:
        for i in tickers:
            print("Dropped " + i)
            data = data[data['Ticker'] != i]
    return data

def readDataSet():
    """
    A function that reads in a dataset. It cleans the data set and standardises it.
    """
    print("Please choose one of the following")
    print("BIS = Istanbul")
    print("BOV = Bovespa")
    print("EUR = EuroStoxx")
    print("JSE = JSE")
    print("NAS = NASDAQ")
    print("SP5 = S&P")
    name = input()
    if name == "BIS":
        data = pd.read_csv("./Data Sets/Cleaned Data Sets/Standardised/cleanedAndEarlyStartBist50.csv")
        data.drop(data.columns[0], axis=1, inplace=True)
        # Following tickers need to be dropped - after this earliest start is 2013
        data = data[data['Ticker'] != "SOKM.IS"]
        data = data[data['Ticker'] != "PGSUS.IS"]
        data = data[data['Ticker'] != "MPARK.IS"]
        data = data[data['Ticker'] != "ENJSA.IS"]
        data = createUniformStart(data)
        # Following tickers need to be dropped, based on shareprice having negative values.
        data = enforcePositiveClose(data)
        print("Finished standardisation of data - proceed with tests")
        return data
    elif name == "BOV":
        data = pd.read_csv("./Data Sets/Cleaned Data Sets/Standardised/bovespaStandardised.csv")
        data.drop(data.columns[0], axis=1, inplace=True)
        # Nothing to be dropped based on the start date - earliest is from 2009
        data = createUniformStart(data)
        # Following tickers need to be dropped, based on shareprice having negative values. (Nothing dropped)
        data = enforcePositiveClose(data)
        print("Finished standardisation of data - proceed with tests")
        return data
    elif name == "EUR":
        data = pd.read_csv("./Data Sets/Cleaned Data Sets/Standardised/euroStoxx50Standardised.csv")
        data.drop(data.columns[0], axis=1, inplace=True)
        # Nothing to be dropped based on the start date - earliest is from 2005
        data = createUniformStart(data)
        # Following tickers need to be dropped, based on shareprice having negative values. (Dropped - ABI.BR)
        data = enforcePositiveClose(data)
        print("Finished standardisation of data - proceed with tests")
        return data
    elif name == "JSE":
        data = pd.read_csv("./Data Sets/Cleaned Data Sets/Standardised/jseStandardised.csv")
        data.drop(data.columns[0], axis=1, inplace=True)
        # Nothing to be dropped based on the start date - earliest is from 2009
        data = createUniformStart(data)
        # Following tickers need to be dropped, based on shareprice having negative values. (Dropped - RMH.JO)
        data = enforcePositiveClose(data)
        print("Finished standardisation of data - proceed with tests")
        return data
    elif name == "NAS":
        data = pd.read_csv("./Data Sets/Cleaned Data Sets/Standardised/nas50Standardised.csv")
        data.drop(data.columns[0], axis=1, inplace=True)
        # Nothing to be dropped based on the start date - earliest is from 2010
        data = createUniformStart(data)
        # Following tickers need to be dropped, based on shareprice having negative values. (Nothing dropped)
        data = enforcePositiveClose(data)
        print("Finished standardisation of data - proceed with tests")
        return data
    elif name == "SP5":
        data = pd.read_csv("./Data Sets/Cleaned Data Sets/Standardised/Sp50Standardised.csv")
        data.drop(data.columns[0], axis=1, inplace=True)
        # Nothing to be dropped based on the start date - earliest is from 2009
        data = createUniformStart(data)
        # Following tickers need to be dropped, based on shareprice having negative values. (Nothing dropped)
        data = enforcePositiveClose(data)
        print("Finished standardisation of data - proceed with tests")
        return data
    
    print("Input '" + name + "' is not valid, please try again")
    return readDataSet()

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

def anticor(data):
    """
    Preform the anticor trading strategy on a given dataset.
    The hyper parameter is the window size.
    """
    window = 5
    startDate = data.Date.min()
    startPrices = data[data['Date'] == startDate]
    numStocks = len(np.unique(startPrices.Ticker.to_numpy()))
    tick = np.unique(startPrices.Ticker.to_numpy())[0]
    # print("Ticker ")
    # print(tick)
    dates = data[data['Ticker'] == tick]
    # print(dates)
    dates = dates.Date.to_numpy()
    propPort = 1 / numStocks
    # getting the init prop of a stock we can own
    # this will assume that this will come out in the right order - which it should
    propOwnedStart = propPort / startPrices.Close.to_numpy()
    currPort = propOwnedStart
    portHist = np.empty((numStocks,len(dates)))
    # print(portHist.shape)
    for i in range(len(dates)):
        currPort = doAnticorDay(i, dates, data, window, numStocks, currPort)
        currPort = np.array(currPort)
        currPort = currPort.reshape((numStocks,1))
        for j in range(numStocks):
            portHist[j][i] = currPort[j]
        # print(portHist.shape)


def generateLX(type, window, day, dates, data, numStocks):
    """
    Based on equation two from the paper by Borodin, El-Yaniv and Gogan. 
    Implementation of this equation for both LX1 and LX2 depending on the type passed.
    The day passed should be an index date and the dates passed should be the full set of dates - it will make use of what is required.
    It should be noted that this requires us to be in the state where an anticor day is possible (i.e where day - 2*window + 1 >= 0)
    """
    if type == 1:
        start = day - 2*window + 1
        count = 0
        LX1 = np.empty((window,numStocks))
        while start + count < day - window:
            currDay = data[data['Date'] == dates[start + count]]
            #assuming the order is kept - which it seems to by nature
            currDay = currDay.Close.to_numpy()
            LX1[count][:] = np.log(currDay)
            count += 1
        return LX1
    elif type == 2:
        start = day - window + 1
        count = 0
        LX2 = np.empty((window,numStocks))
        #get up to the current day
        # print(day)
        # print(start + count)
        while start + count <= day:
            # print("Count in LX2 is " + str(count))
            currDay = data[data['Date'] == dates[start + count]]
            #assuming the order is kept - which it seems to by nature
            currDay = currDay.Close.to_numpy()
            LX2[count][:] = np.log(currDay)
            count +=1
        return LX2
    else:
        print("Error")
        return -1

def calcMCov(LX1, LX2, numStocks, window):
    """
    This is based on equation 3 from the paper by Borodin, El-Yaniv and Gogan.
    """
    u1 = averageValuesAnticor(LX1)
    u2 = averageValuesAnticor(LX2)
    MCov = np.empty((numStocks, numStocks))
    #Let i index Lx1
    for i in range(numStocks):
        #Let j index Lx2
        for j in range(numStocks):
            MCov[i][j] = ((LX1[:,i] - u1[i]) @ (LX2[:,j] - u2[j]))/(window - 1)
    sigma1 = standardDevAnticor(LX1)
    sigma2 = standardDevAnticor(LX2)
    print(sigma1.shape)
    print(sigma2.shape)
    MCor = np.empty((numStocks, numStocks))
    for i in range(numStocks):
        for j in range(numStocks):
            if sigma1[i] != 0:
                if sigma2[j] != 0:
                    MCor[i][j] = MCov[i][j] / (sigma1[i] * sigma2[j])
                else: 
                    MCor[i][j] = 0
            else:
                MCor[i][j] = 0
    
    # Following serves to check that the required criteria - that MCor is always between -1 and 1
    flattenedCor = MCor.flatten()
    errorsPos = np.where(flattenedCor > 1)[0]
    if len(errorsPos) > 0:
        print("Number of mistakes greater than 1 is " + str(len(errorsPos)))
        print("Mistakes - greater than 1")
        print(flattenedCor[errorsPos])
    errorsNeg = np.where(flattenedCor < -1)[0]
    if len(errorsNeg) > 0:
        print("Number of mistakes less than 1 is " + str(len(errorsNeg)))
        print("Mistakes - less than 1")
        print(flattenedCor[errorsNeg])
    if(len(errorsNeg) == 0 and len(errorsPos) == 0):
        print("Success in calculating Correlation")
        return MCor, u2
    # If some error occurred then return -1, so catch this error somewhere
    return -1, u2
def averageValuesAnticor(LXk):
    """
    Given an LX, find the average values for each column and return them.
    This only exists as a function to make it a bit more obvious what is being done inside other code segments
    """
    return np.mean(LXk, axis=0)

def standardDevAnticor(LXk):
    """
    Given an LX, find the average standard deviation for each column and return them.
    This only exists as a function to make it a bit more obvious what is being done inside other code segments
    """
    return np.std(LXk, axis=0)

def claimItoJ(MCor, i, j):
    """
    Computing the claim - which represents the amount of our investment we wish to shift from asset i to asset j.
    Ensure that the requirements are met before running this function which are as follows:
    1) MCor(i,j) > 0
    2) u2(i) > u2(j)
    This function will assume that these two criteria have been checked first and validated
    """


    return

def doAnticorDay(day, dates, data, window, numStocks, currPort):
    """
    Given the parameters of anticor, preform a single anticor trading day.
    Using the current portfolio change it as it is required and return a new portfolio.
    Note that here day is the current day, dates the array of all days, data is the exchange, numStocks must be passed, currPort will be updated.
    """
    # Case where not enough data
    if day < 2*window:
        return currPort
    # Enough data so start strategy
    lx1 = generateLX(1, window, day, dates, data, numStocks)
    lx2 = generateLX(2, window, day, dates, data, numStocks) 
    # Generating MCov as required
    MCov, u2 = calcMCov(lx1, lx2, numStocks, window)
    if MCov == -1:
        print("An error occured in calcMCov, where a value was out of bounds")
        return -1
    # Calculate claims - assuming above works
    
    # Deal with case of u2
    
    # Create new portfolio based on claims and transfers

data = readDataSet()
anticorReturns = anticor(data)