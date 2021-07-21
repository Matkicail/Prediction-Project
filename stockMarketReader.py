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