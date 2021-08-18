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
name = "BOV600"
startDate = 2500
endDate = 2650
# bestStock, stockName = np.log(bestStockStrategy(data))
# ubahPort = np.log(ubah(data))
# crpPort = np.log(UCRP(data))
# # bestStock = np.log(bestStockStrategy(data))
# cornReturns = np.log(np.loadtxt("./Data Sets/CORNK/" + name + "DAYCORNRETURNS.txt"))
# racornReturns = np.log(np.loadtxt("./Data Sets/RACORNK/" + name + "DAYRACORNRETURNS.txt"))
# dricornReturns = np.log(np.loadtxt("./Data Sets/DRICORNK/" + name + "DAYDRICORNRETURNS.txt"))
# weirdkReturns= np.load(np.load("./Data Sets/WEIRDK/" + name + ".txt"))

bestStock, stockName = bestStockStrategy(data, startDate, endDate)
ubahPort = ubah(data, startDate, endDate)
crpPort = UCRP(data, startDate, endDate)
# cornReturns = np.loadtxt("./Data Sets/CORNK/" + name + "DAYCORNRETURNS.txt")
# racornReturns = np.loadtxt("./Data Sets/RACORNK/" + name + "DAYRACORNRETURNS.txt")
# dricornReturns = np.loadtxt("./Data Sets/DRICORNK/" + name + "DAYDRICORNRETURNS.txt")
# weirdkReturns= np.loadtxt("./Data Sets/WEIRDK/" + name + ".txt")

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



numDays = len(bestStock)

# specialShowCase50 = np.loadtxt("./bullShowcaseJse.txt")
# specialShowCase20 = np.loadtxt("./bullShowcaseJseTrainSize20.txt")
specialShowCase10 = np.loadtxt("./bullShowcaseJseTrainSize10.txt")
numDays = len(specialShowCase10)
if len(crpPort) < numDays:
    numDays = len(crpPort)

plt.title("Comparison of Strategies")
plt.xlabel("Number of Days")
plt.ylabel("Total Return")
plt.plot(bestStock[0:numDays], label=stockName)
plt.plot(ubahPort[0:numDays], label="UBAH")
plt.plot(crpPort[0:numDays], label="CRP")
plt.plot(specialShowCase10[1:numDays], label="WEIRDK-10")
# plt.plot(specialShowCase20[0:numDays], label="WEIRDK-20")
# plt.plot(specialShowCase50[0:numDays], label="WEIRDK-50")
# plt.plot(cornReturns, label="CORN")
# plt.plot(racornReturns, label="RACORN")
# plt.plot(dricornReturns, label="DRICORN")
# plt.plot(weirdkReturns, label="Capital Gains Bot")
plt.legend()
plt.show()