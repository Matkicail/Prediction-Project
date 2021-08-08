from datetime import date
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from stockMarketReader import readDataSet
from bestStock import bestStockStrategy
from ubah import ubah
from CRP import UCRP


data = readDataSet()
# bestStock = np.log(bestStockStrategy(data))
ubahPort = np.log(ubah(data))
crpPort = np.log(UCRP(data))

# bestStock = bestStockStrategy(data)
# ubahPort = ubah(data)
# crpPort = UCRP(data)
cornReturns = np.log(np.loadtxt("TEMPCORNRETURNS.txt"))
numDays = len(cornReturns)
plt.title("Comparison of Strategies")
plt.xlabel("Number of Days")
plt.ylabel("Total Return")
# plt.plot(bestStock, label="Best Stock")
plt.plot(ubahPort[0:numDays], label="UBAH")
plt.plot(crpPort[0:numDays], label="CRP")
plt.plot(cornReturns, label="CORN")
plt.legend()
plt.show()