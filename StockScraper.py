import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas_datareader as pdr
import pandas as pd
import datetime as dt
import os

class StockDownloader:
    def __init__(self, url = "https://sashares.co.za/jse-top-40/", topStocksIDs=None, tryCache=True):

        #Set up paramters
        self.url = url
        self.topStocksIDs = topStocksIDs
        self.topStocksData = None
        self.cleanedData = None
        self.priceRelativeData = None
        self.tryCache = tryCache

        #Names for directory and files
        self.cachedDirectory = "StockData"
        self.cachedFile = "stocks"
        self.cleanedCachedFile = "cleaned"
        self.relativeFile = "relative"

        #Full paths from names
        self.fullPath = "{0}/{1}.zip".format(self.cachedDirectory, self.cachedFile)
        self.cleanFullPath = "{0}/{1}_{2}.zip".format(self.cachedDirectory, self.cleanedCachedFile, self.cachedFile)
        self.relativeFullPath = "{0}/{1}_{2}.zip".format(self.cachedDirectory, self.cleanedCachedFile, self.cachedFile)


        #Compression config
        self.compressionConfig = {'method' : 'zip', 'archive_name' : self.cachedFile + ".csv"}


        #Tries to see if we have downloaded data
        if tryCache:
            if not self.LoadCachedData():
                self.DownloadTopStocks()
        else:
            self.DownloadTopStocks()
        

    def LoadCachedData(self):
        """
        Function to load Cached data. Will run by default unless specified not to.
        """
        if(self.tryCache == False):
            print("Not allowed to load cache")
            return False

        #If not a directory, create it
        if (not os.path.isdir(self.cachedDirectory)):
            os.makedirs(self.cachedDirectory)
            return False
        else:
            #TODO could be improved
            foundData = False
            if(os.path.isfile(self.fullPath)):
                self.topStocksData = pd.read_csv(self.fullPath, dtype=object).to_numpy()
                foundData = True
            
            if(os.path.isfile(self.cleanFullPath)):
                self.cleanedData = pd.read_csv(self.cleanFullPath, dtype=object).to_numpy()

            if(os.path.isfile(self.relativeFullPath)):
                self.priceRelativeData = pd.read_csv(self.relativeFullPath, dtype=object).to_numpy()

            return foundData


    def DownloadTopStocksIDs(self):
        """
        Scrapes to top 40 JSE stocks. Note, only works with one website https://sashares.co.za/jse-top-40/ so far
        """
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
    
    def DownloadData(self):
        """
        A helper function that will automate the downloading of data, will clean it, and get price relatives
        """
        self.DownloadTopStocks()
        self.CleanData()
        self.CalculatePriceRelative()

    #Download all top stocks from beginning
    def DownloadTopStocks(self):
        if self.topStocksIDs is None:
            self.DownloadTopStocksIDs()
        
        assert self.topStocksIDs is not None, "Error, Stock IDs are missing."

        #Get Dates and download data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.today()

        #Download stock
        topStocksData = pdr.get_data_yahoo(self.topStocksIDs, start, end)
        self.topStocksData = topStocksData.to_numpy()
        
        #Save
        topStocksData.to_csv(self.fullPath, compression=self.compressionConfig)

        return self


    def CleanData(self, thresholdRowAmount = None):

        #Data loaded from csv is different from downloaded data
        assert self.LoadCachedData(), "Can only clean data on cache loaded data"

        data = self.topStocksData

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

        #Insert Dates
        data = np.insert(data, 0, dates, axis=1)

        #Insert symbols
        symbols[0]="Date"
        data = np.insert(data, 0, symbols, axis=0)

        self.cleanedData = data.copy()

        #Rebuild pandas to save
        df = pd.DataFrame(data)
        df.to_csv(self.cleanFullPath, compression=self.compressionConfig)

        return self

    def CalculatePriceRelative(self):

        assert self.cleanedData is not None, "Requires cleaned data to run"

        #Extract cleaned data
        data = self.cleanedData.copy()[1:,1:].astype(float)

        #Calculate price relatives and insert ones
        priceRelative = data[1:,:] / data[:-1,:]
        priceRelative= np.insert(priceRelative, 0, np.ones((1,priceRelative.shape[1])), axis=0)

        #Add headings and dates
        priceRelative = priceRelative.astype(object)
        priceRelative = np.insert(priceRelative, 0, self.cleanedData[1:,0], axis=1)
        priceRelative = np.insert(priceRelative, 0, self.cleanedData[0,:], axis=0)

        #Copy and save
        self.priceRelativeData = priceRelative.copy()

        #Save
        df = pd.DataFrame(priceRelative)
        df.to_csv("{0}/{1}_{2}.zip".format(self.cachedDirectory, self.relativeFile, self.cachedFile), compression=self.compressionConfig)

        return self

    def UpdateTopStocks(self):
        print("TODO, Implement")
        pass


a = StockDownloader(tryCache=True)
a.DownloadData()
