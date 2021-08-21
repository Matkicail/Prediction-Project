import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas_datareader as pdr
import pandas as pd
import datetime as dt
import os

class StockDownloader:
    def __init__(self, url = "https://sashares.co.za/jse-top-40/", topStocksIDs=None, tryCache=True):
        self.url = url
        self.topStocksIDs = topStocksIDs
        self.topStocksData = None

        self.cachedDirectory = "StockData"
        self.cachedFile = "stocks"
        self.cleanedCachedFile = "cleaned"
        self.fullPath = "{0}/{1}.zip".format(self.cachedDirectory, self.cachedFile)

        self.compressionConfig = {'method' : 'zip', 'archive_name' : self.cachedFile + ".csv"}

        self.tryCache = tryCache

        self.cleanedNumpyData = None

        #Tries to see if we have downloaded data
        if tryCache:
            if not self.LoadCachedData():
                self.DownloadTopStocks()
        else:
            self.DownloadTopStocks()
        

    def LoadCachedData(self):

        if(self.tryCache == False):
            print("Not allowed to load cache")
            return False

        #If not a directory, create it
        if (not os.path.isdir(self.cachedDirectory)):
            os.makedirs(self.cachedDirectory)
            return False
        else:
            if(os.path.isfile(self.fullPath)):
                self.topStocksData = pd.read_csv(self.fullPath, dtype=object)
                return True
        return False


    def DownloadTopStocksIDs(self):

        #Get page and soup it
        page = requests.get(self.url)
        soup = BeautifulSoup(page.content, "html.parser")

        #Find table rows
        rows = soup.find(id="table_1").find("tbody").find_all("tr")

        #Get stock ids
        self.topStocksIDs = []
        for row in rows:
            self.topStocksIDs.append(row.extract().next_element.next_element.findNext(text=True) + ".JO")

        return self

    #Download all top stocks from beginning
    def DownloadTopStocks(self, cacheFile = True):
        if self.topStocksIDs is None:
            self.DownloadTopStocksIDs()
        
        assert self.topStocksIDs is not None, "Error, Stock IDs are missing."

        #Get Dates and download data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.today()

        self.topStocksData = pdr.get_data_yahoo(self.topStocksIDs, start, end)

        #Cache it if possible
        if cacheFile:
            self.topStocksData.to_csv(self.fullPath, compression=self.compressionConfig)

        return self

    def UpdateTopStocks(self):
        pass

    def CleanData(self, thresholdRowAmount = None):

        assert self.LoadCachedData(), "Can only clean data on cache loaded data"

        #Data loaded from csv is different from downloaded data
        data = self.topStocksData.to_numpy()
        #Read data
        data = np.delete(data, 1, 0)
        symbols = data[0,0:41]
        dates = data[1:,0]
        data = data[1:,1:41].astype(np.float)
        
        #Find the max non nan number row
        row = np.where(np.isnan(data))[0]

        #Determine if we should threshold to keep more data
        if thresholdRowAmount is not None:
            row = row[row <= thresholdRowAmount]
        
        #Extract data
        row = np.max(row) + 1
        data = data[row:, :].astype(np.object)
        dates = dates[row:]

        self.cleanedNumpyData = data.copy()

        #Insert Dates
        data = np.insert(data, 0, dates, axis=1)

        #Insert symbols
        symbols[0]="Date"
        data = np.insert(data, 0, symbols, axis=0)

        #Rebuild pandas to save
        df = pd.DataFrame(data)
        df.to_csv("{0}/{1}_{2}.zip".format(self.cachedDirectory, self.cleanedCachedFile, self.cachedFile), compression=self.compressionConfig)

        self.CalculatePriceRelative()

    def CalculatePriceRelative(self):
        data = self.cleanedNumpyData

        priceRelative = data[1:,:] / data[:-1,:]
        priceRelative= np.insert(priceRelative, 0, np.ones((1,priceRelative.shape[1])), axis=0)
        self.priceRelative = priceRelative

a = StockDownloader(tryCache=True)
a.CleanData()
