import yfinance as yf
import numpy as np
import pandas as pd

##
# @method a method to generate stock csvs for a given exchange using stock indices
# @description create a csv of a set of stocks showing their performance over time
# @param name this is the name we want the csv to have
# @param tickers these are the tickers we want info on
# @returns void
##
def generateStockCSV(name,tickers):
    standardCols = pd.DataFrame(columns=["Date","Ticker","Open","High","Low","Close","Volume","Dividends","Stock Splits","Mkt Cap"])
    stockInfo = np.empty(())
    
    for i in tickers:
        print("######################")
        print("getting info for " + i)
        print("######################")
        stockInfo = yf.Ticker(i).history(period="max")
        stockInfo.insert(loc=0, column="Ticker", value=np.repeat(i,len(stockInfo)))
        stockInfo.reset_index(inplace=True)
        standardCols = standardCols.append(stockInfo, ignore_index=True)
        
    standardCols.to_csv(name+".csv")

# JSE TOP 40
jseStock = ["BTI.JO", "NPN.JO", "CFR.JO", "AGL.JO", "SOL.JO", "SBK.JO", "VOD.JO",
            "MND.JO", "MNP.JO", "SLM.JO", "MTN.JO", "NED.JO", "AMS.JO", "DSY.JO",
            "SHP.JO", "RMH.JO", "INP.JO", "REM.JO", "INL.JO", "KIO.JO", "APN.JO",
            "GRT.JO", "BVT.JO", "RMI.JO", "WHL.JO", "ANG.JO", "TBS.JO", "REI.JO",
            "CCO.JO", "NTC.JO", "LHC.JO", "IPL.JO", "IMP.JO", "FSR.JO", "GFI.JO",
            "EXX.JO", "CLS.JO", "CPI.JO", "MRP.JO", "NHM.JO", "SPP.JO"]

# S&P 500 TOP 50 - these are in order
SP500Stock = ["AAPL", "MSFT", "AMZN", "FB", "GOOGL", "GOOG", "BRK-B", "TSLA",
                "NVDA", "JPM", "JNJ", "V", "UNH", "PYPL", "HD", "MA", "DIS",
                "BAC", "ADBE", "XOM", "CMCSA", "NFLX", "VZ", "INTC", "CRM",
                "CSCO", "PFE", "KO", "ABT", "T", "PEP", "CVX", "ABBV",
                "TMO", "NKE", "MRK", "AVGO", "WMT", "ACN", "WFC", "LLY", "TXN",
                "COST", "DHR", "MDT", "QCOM", "PM", "HON"]

# Nasdaq TOP 50 - these are in order
nasdaqStock = ["AAPL", "MSFT", "AMZN", "GOOG", "FB", "NVDA", "TSLA", "GOOGL",
                "PYPL", "ADBE", "CMCSA", "NFLX", "INTC", "CSCO", "PEP", "AVGO",
                "TMUS", "COST", "TXN", "QCOM", "CHTR", "AMGN", "INTU", "SBUX",
                "AMAT", "ISRG", "ZM", "MRNA", "BKNG", "MDLZ", "MU", "LRCX",
                "ADP", "GILD", "MELI", "CSX", "FISV", "ATVI", "ILMN", "ADSK",
                "REGN", "ADI", "JD", "IDXX", "DOCU", "ASML", "NXPI", "BIIB"]

#  Bovespa TOP 30 - some really weird stuff in here, it is in the inverse order as well btw
bovespaStock = ["PETR4.SA", "ITUB3.SA", "BBDC3.SA", "ABEV3.SA", "VALE3.SA",
                "SANB11.SA", "BBAS3.SA", "ITSA4.SA", "VIVT3.SA", "B3SA3.SA",
                "JBSS3.SA", "ELET3.SA", "WEGE3.SA", "EGIE3.SA", "LREN3.SA",
                "CPFE3.SA", "SBSP3.SA", "RAIL3.SA", "RENT3.SA", "CCRO3.SA",
                "BRKM5.SA", "TIMS3.SA", "BRFS3.SA", "LAME4.SA", "PCAR3.SA",
                "GGBR4.SA", "NTCO3.SA", "CMIG4.SA", "CSNA3.SA", "UGPA3.SA"]

#  EuroStoxx top 50 - need to ensure only EU and all in EUR / they aren't in order and can safely go to ~2000 year
# Quite a few issues here to solve - mainly ensure that it is all in the right index
euroStoxx50 = ["AI.PA", "ALV.DE", "ABI.BR", "MT.AS", "ASML.AS", "G.MI", "CS.PA",
                "BBVA.MC", "SAN.MC", "BAS.DE", "BAYN.DE", "BMW.DE", "BNP.PA",
                "CA.PA", "SGO.PA", "CRH.L", "DAI.DE", "BN.PA", "DBK.DE", "DTE.DE",
                "ENEL.MI", "ENGI.PA", "ENI.MI", "EOAN.DE", "EL.PA", "IBE.MC", "ITX.MC",
                "INGA.AS", "ISP.MI", "PHIA.AS",  "OR.PA", "MC.PA", "MUV2.DE", "NOKIA.HE",
                "ORA.PA", "REP.MC", "RWE.DE", "SAN.PA", "SU.PA", "SIE.DE", "GLE.PA",
                "TEF.MC", "TTE.PA", "UCG.MI", "UNA.AS", "DG.PA", "VIV.PA", "VOW.DE"]

# BIST 50 - Turkish top 50 stock exchange
borsaStock = ["AKBNK.IS", "ALARK.IS", "ALKIM.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "CCOLA.IS",
                "DOHOL.IS", "EKGYO.IS", "ENJSA.IS", "EREGL.IS", "FENER.IS", "FROTO.IS", "GARAN.IS",
                "GUBRF.IS", "HEKTS.IS", "IHLAS.IS", "IPEKE.IS", "ISFIN.IS", "KRDMD.IS", "KARTN.IS",
                "KCHOL.IS", "KOZAL.IS", "KOZAA.IS", "MGROS.IS", "MPARK.IS", "ODAS.IS", "OTKAR.IS",
                "OYAKC.IS", "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SELEC.IS", "SISE.IS",
                "SOKM.IS", "TAVHL.IS", "TKFEN.IS", "THYAO.IS", "TSKB.IS", "TUPRS.IS", "TTKOM.IS",
                "TCELL.IS", "HALKB.IS", "ISCTR.IS", "TURSG.IS", "ULKER.IS", "VAKBN.IS", "VESTL.IS",
                "YKBNK.IS"
            ]
#need to try and download this but not necessary right now.
nikkeiStock = ["7203.T", "6861.T", "6758.T", "9984.T", "9432.T", "6098.T", "9433.T", "9983.T",
                "8306.T", "6594.T", "4063.T", "7974.T", "8035.T", "4519.T", "6367.T", "6501.T",
                "7267.T", "4502.T", "6902.T", "6981.T", "7741.T", "8316.T", "4661.T", "8001.T",
                "6954.T", "3382.T", "6273.T", "4689.T", "8316.T", "8411.T", "6702.T", "4568.T",
                "2914.T", "8766.T", "4503.T", "7182.T", "6178.T", "5108.T", "4612.T", "4543.T",
                "6503.T", "4901.T", "4452.T", "9022.T", "6752.T", "4911.T", "6201.T", "9020.T",
                "7733.T", "6326.T"
            ]

# print("Generating JSE")
# generateStockCSV("jse40",jseStock)
print("Generating S&P 500")
generateStockCSV("sp50",SP500Stock)
# print("Generating Nasdaq")
# generateStockCSV("nas50",nasdaqStock)
# print("Generating EuroStoxx")
# generateStockCSV("euroStoxx50",euroStoxx50)
# print("Generating Borsa")
# generateStockCSV("bist50",borsaStock)
# print("Generating Bovespa")
# generateStockCSV("bovespa30",bovespaStock)
# print("Finished Generation")