from datetime import date
import numpy as np
from numpy.lib import corrcoef
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import data
from stockMarketReader import readDataSet
import math
from cornk import cornDataRead
from cornk import boundsCreator
from cornk import constraintSumOne
from cornk import initialGuess

def R(b, bCurr):
    """
    Regularising term
    """
    pass

def objective(portfolio, bCurr ,day, setSize, ita):
    """
    Ensuring that days is a vector/matrix where width is number of days and length is numStocks.
    """

    total = ita * np.log(portfolio @ day) - R(b,bCurr)

    # Return negative portfolio so that we can minimise (hence maximising the portfolio)
    return -total


def runExponential(dataSet, startDateEarly, startDate):
    """
    Run the exponential gradient algorithm
    """
    pass





trainSize = 100
startDate = 200
startDateEarly = startDate - trainSize
endDate = 708
market = input("Which market ? \n")
dataSet = cornDataRead()
dataSet = dataSet[startDateEarly:endDate+1]

