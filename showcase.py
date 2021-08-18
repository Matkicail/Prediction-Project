<<<<<<< HEAD
from datetime import date
from re import I
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
startDate = 2900
endDate = 3193
# bestStock, stockName = np.log(bestStockStrategy(data))
ubahPort = np.log(ubah(data, startDate, endDate))
crpPort = np.log(UCRP(data, startDate, endDate))
# bestStock, stockName = np.log(bestStockStrategy(data, startDate, endDate))
# cornReturns = np.log(np.loadtxt("./Data Sets/CORNK/" + name + "DAYCORNRETURNS.txt"))
# racornReturns = np.log(np.loadtxt("./Data Sets/RACORNK/" + name + "DAYRACORNRETURNS.txt"))
# dricornReturns = np.log(np.loadtxt("./Data Sets/DRICORNK/" + name + "DAYDRICORNRETURNS.txt"))
# weirdkReturns= np.load(np.load("./Data Sets/WEIRDK/" + name + ".txt"))

bestStock, stockName = bestStockStrategy(data, startDate, endDate)
bestStock = np.log(bestStock)
# ubahPort = ubah(data, startDate, endDate)
# crpPort = UCRP(data, startDate, endDate)
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



numDays = endDate - startDate
datesInterest = np.arange(start = 20, stop=endDate-startDate, step=10)
specialShowCase10 = np.loadtxt("./2900-3200bulljseTrainSize10.txt")
specialShowCase20 = np.loadtxt("./2900-3200bulljseTrainSize20.txt")
specialShowCase30 = np.loadtxt("./2900-3200bulljseTrainSize30.txt")

specialShowCase10 = np.log(specialShowCase10)
specialShowCase20 = np.log(specialShowCase20)
specialShowCase30 = np.log(specialShowCase30)

# numDays = len(specialShowCase10)
for i in datesInterest:

    bestStock, stockName = bestStockStrategy(data, startDate, startDate + i)
    numDays = endDate - startDate + i
    plt.figure(figsize=(10,10))
    bestStock = np.log(bestStock)
    plt.title("Comparison of Strategies")
    plt.xlabel("Number of Days")
    plt.ylabel("Total Return")
    plt.plot(bestStock[0:i], label=stockName)
    plt.plot(ubahPort[0:i], label="UBAH")
    plt.plot(crpPort[0:i], label="CRP")
    plt.plot(specialShowCase10[1:i], label="WEIRDK-10")
    plt.plot(specialShowCase30[1:i], label="WEIRDK-20")
    plt.plot(specialShowCase20[1:i], label="WEIRDK-30")
    # plt.plot(cornReturns, label="CORN")
    # plt.plot(racornReturns, label="RACORN")
    # plt.plot(dricornReturns, label="DRICORN")
    # plt.plot(weirdkReturns, label="Capital Gains Bot")
    plt.legend()
    plt.show()
=======
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
>>>>>>> 1b01d6c536f027c2eecf17ce4f790f307377dff4
