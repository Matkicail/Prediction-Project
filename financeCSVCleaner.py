import pandas as pd

def cleanCSV(nameCSV, cleanedName):
    dirtyCSV = pd.read_csv(nameCSV)
    colsToDrop = ["Unnamed: 0","Open", "High", "Low", "Volume", "Dividends", "Stock Splits", "Mkt Cap"]
    dirtyCSV = dirtyCSV.drop(colsToDrop, axis=1)
    dirtyCSV.to_csv(cleanedName+".csv")


cleanCSV("bist50.csv","cleanedBist50")
cleanCSV("jse40.csv","cleanedJse40")
cleanCSV("bovespa30.csv","cleanedBovespa30")
cleanCSV("euroStoxx50.csv","cleanedEuroStoxx50")
cleanCSV("nas50.csv","cleanedNas50")
cleanCSV("sp50.csv","cleanedSp50")