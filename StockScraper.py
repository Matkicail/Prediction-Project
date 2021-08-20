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
        self.cachedFile = "stocks.zip"
        self.fullPath = "{0}/{1}".format(self.cachedDirectory, self.cachedFile)

        #Tries to see if we have downloaded data
        if tryCache:
            if not self.LoadCachedData():
                self.DownloadTopStocks()
        else:
            self.DownloadTopStocks()
        

    def LoadCachedData(self):
        #If not a directory, create it
        if (not os.path.isdir(self.cachedDirectory)):
            os.makedirs(self.cachedDirectory)
            return False
        else:
            if(os.path.isfile(self.fullPath)):
                self.topStocksData = pd.read_csv(self.fullPath, compression='gzip')
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

        #self.topStocksData.drop(self.topStocksData.columns[40:])

        #Cache it if possible
        if cacheFile:
            self.topStocksData.to_csv(self.fullPath, compression='gzip')

        return self

    def UpdateTopStocks(self):
        pass

    def CleanData(self, thresholdRowAmount = None):

        #Read data
        data = self.topStocksData.to_numpy()
        symbols = np.unique(data[0,1:])
        dates = data[1:,0]
        data = data[1:,1:].astype(np.float)
        
        #Find the max non nan number row
        row = np.where(np.isnan(data))[0]

        if thresholdRowAmount is not None:
            row = row[row <= thresholdRowAmount]
        
        row = np.max(row) + 1
        data = data[row:, :]
        b=2



a = StockDownloader(tryCache=False)
a.CleanData()
