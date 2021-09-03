from datetime import date
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from stockMarketReader import readDataSet
from bestStock import bestStockStrategy
from ubah import ubah
from CRP import UCRP
from scipy import stats

data = readDataSet()

startDate = 908
endDate = 1124
numDays = endDate  -startDate
logged = False
bestStock, stockName = bestStockStrategy(data, startDate, endDate)
ubahPort = ubah(data, startDate, endDate)
crpPort = UCRP(data, startDate, endDate)

market = input("Input Market Name \n")
# mixedModel = np.loadtxt("./Data Sets/MIXEDMODEL-Test-start-1500-end-2324-JSE-sizesmixed-model-10-120.txt")
# cornReturns = np.loadtxt("./Data Sets/CORNK/{0}-Exchange-StartDate908-EndDate1162.txt".format(market))
# racornReturns = np.loadtxt("./Data Sets/RACORNK/{0}-Exchange-StartDate908-EndDate1162.txt".format(market))
# weirdkReturnsSmall= np.loadtxt("./Data Sets/Test-SIMPLE-start-898-end-1162-{0}-sizes-10.txt".format(market))
# weirdkReturnsMedium = np.loadtxt("./Data Sets/Test-SIMPLE-start-788-end-1162-{0}-sizes-120.txt".format(market))
# weirdKReturnsLarge = np.loadtxt("./Data Sets/Test-SIMPLE-start-718-end-1162-{0}-sizes-190.txt".format(market))
# dricornReturns = np.loadtxt("./Data Sets/DRICORNK/" + name + "DAYDRICORNRETURNS.txt")
# weirdkReturns= np.loadtxt("./Data Sets/WEIRDK/" + name + ".txt")


# bestStock = np.log(bestStock)
# ubahPort = np.log(ubahPort)
# crpPort = np.log(crpPort)
# cornReturns = np.log(cornReturns)
# racornReturns = np.log(racornReturns)
# weirdkReturnsSmall = np.log(weirdkReturnsSmall)
# weirdkReturnsMedium = np.log(weirdkReturnsMedium)
# weirdKReturnsLarge = np.log(weirdKReturnsLarge)
# logged = True

# box-cox transform showcase
# bestStock, stockName = bestStockStrategy(data)
# ubahPort = ubah(data)
# crpPort = UCRP(data)
# cornReturns = np.loadtxt("./Data Sets/CORNK/" + name + "DAYCORNRETURNS.txt")
# racornReturns = np.loadtxt("./Data Sets/RACORNK/" + name + "DAYRACORNRETURNS.txt")
# dricornReturns = np.log(np.loadtxt("./Data Sets/DRICORNK/" + name + "DAYDRICORNRETURNS.txt"))
# numDays = len(cornReturns)
# bestStock, bestStock_lambda = stats.boxcox(bestStockStrategy(data[0:numDays]))
# ubahPort, ubah_lambda = stats.boxcox(ubahPort[0:numDays])
# crpPortPort, crpPort_lambda = stats.boxcox(crpPort[0:numDays])
# cornReturns, cornReturns_lambda = stats.boxcox(cornReturns)
# racornReturns, racornReturns_lambda = stats.boxcox(racornReturns)
# dricornReturns , dricornReturns_lambda stats.boxcox(dricornReturns)


if logged == False:
    plt.title("Comparison of Strategies - {0} TESTING DATA {1}-{2}".format(market, startDate, endDate))
if logged == True:
    plt.title("Logged Comparison of Strategies - {0} TESTING DATA {1}-{2}".format(market, startDate, endDate))
plt.xlabel("Number of Days")
plt.ylabel("Total Return")
plt.plot(bestStock[0:numDays], label=stockName)
plt.plot(ubahPort[0:numDays], label="UBAH")
plt.plot(crpPort[0:numDays], label="CRP")

# plt.plot(mixedModel, label="Mixed Model")

# plt.plot(cornReturns[1:numDays], label="CORN")
# plt.plot(racornReturns[0:numDays], label="RACORN")
# plt.plot(weirdkReturnsSmall[1:numDays], label="10-Day")
# plt.plot(weirdkReturnsMedium[1:numDays], label="110-Day")
# plt.plot(weirdKReturnsLarge[1:numDays], label="190-Day")
plt.ylabel("Returns")
plt.xlabel("# Trading Days")
plt.legend()
plt.show()

