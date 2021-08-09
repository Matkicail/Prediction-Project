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
name = "BOV500"
# bestStock = np.log(bestStockStrategy(data))
# ubahPort = np.log(ubah(data))
# crpPort = np.log(UCRP(data))
# bestStock = np.log(bestStockStrategy(data))
# cornReturns = np.log(np.loadtxt("./Data Sets/CORNK/" + name + "DAYCORNRETURNS.txt"))
# racornReturns = np.log(np.loadtxt("./Data Sets/RACORNK/" + name + "DAYRACORNRETURNS.txt"))

# bestStock = bestStockStrategy(data)
# ubahPort = ubah(data)
# crpPort = UCRP(data)
# cornReturns = np.loadtxt("./Data Sets/CORNK/" + name + "DAYCORNRETURNS.txt")
# racornReturns = np.loadtxt("./Data Sets/RACORNK/" + name + "DAYRACORNRETURNS.txt")

# box-cox transform showcase
bestStock = bestStockStrategy(data)
ubahPort = ubah(data)
crpPort = UCRP(data)
cornReturns = np.loadtxt("./Data Sets/CORNK/" + name + "DAYCORNRETURNS.txt")
racornReturns = np.loadtxt("./Data Sets/RACORNK/" + name + "DAYRACORNRETURNS.txt")
numDays = len(cornReturns)
bestStock, bestStock_lambda = stats.boxcox(bestStockStrategy(data[0:numDays]))
ubahPort, ubah_lambda = stats.boxcox(ubahPort[0:numDays])
crpPortPort, crpPort_lambda = stats.boxcox(crpPort[0:numDays])
cornReturns, cornReturns_lambda = stats.boxcox(cornReturns)
racornReturns, racornReturns_lambda = stats.boxcox(racornReturns)

numDays = len(cornReturns)
if len(racornReturns) < numDays:
    cornReturns = cornReturns[0:len(racornReturns)]
plt.title("Comparison of Strategies")
plt.xlabel("Number of Days")
plt.ylabel("Total Return")
plt.plot(bestStock[0:numDays], label="Best Stock")
plt.plot(ubahPort[0:numDays], label="UBAH")
plt.plot(crpPort[0:numDays], label="CRP")
plt.plot(cornReturns, label="CORN")
plt.plot(racornReturns, label="RACORN")
plt.legend()
plt.show()