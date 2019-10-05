import pandas as pd
import numpy as np
from arch import arch_model

import technical as tc

def clean(df):
    '''nan is removed as we have large dataset.
    '''
    if df.isnull().values.any():
        df.dropna(inplace = True)

def fundamental(df):
    '''Append fundamental data
    '''
    df.set_index(df['Date'], inplace = True)
    #save time series for easier accessing
    open = df['Open']
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    ret = df['Close'].pct_change()
    df['HH'] = high.rolling(window = 14).max() #Highest high
    df['LL'] = low.rolling(window = 14).min() #Lowest low
    df['Volume_chg'] = volume.pct_change()
    df['Intra_ret'] = np.log(open/close)
    df['Return_1D'] = ret
    df['Return_3D'] = np.log(close / close.shift(3))
    df['SMA_5D'] = ret.rolling(5).mean()
    df['SMA_13D'] = ret.rolling(13).mean()
    df['SMA_21D'] = ret.rolling(21).mean()
    df['SD_5D'] = ret.rolling(5).std()
    df['SD_13D'] = ret.rolling(13).std()
    df['SD_21D'] = ret.rolling(21).std()
    df['EMA_5D'] = ret.ewm(span = 5).mean()
    df['EMA_13D'] = ret.ewm(span = 13).mean()
    df['EMA _21D'] = ret.ewm(span = 21).mean()
    #Auto_correlation
    df['AutoCorr_1'] = df['Close'].rolling(90).apply(lambda x: x.autocorr(lag = 1), raw = False)
    df['AutoCorr_5'] = df['Close'].rolling(90).apply(lambda x: x.autocorr(lag = 5), raw = False)
    df['AutoCorr_13'] = df['Close'].rolling(90).apply(lambda x: x.autocorr(lag = 13), raw = False)
    #GARCH model
    u = ret.dropna() * 100
    garch = arch_model(u, p = 1, q = 1)
    res = garch.fit()
    garch_vol = np.zeros_like(df['Close'])
    for i in range(2, len(df)):
        garch_vol[i] = res.params[1] + (res.params[2] * (u[i-1]**2)) + (res.params[3] * (garch_vol[i-1]))
    df['GARCH'] = pd.Series(garch_vol, index = df.index.values)

def technical(df):
    '''Technical Indicator.

    Important note: More technical data will be included. It will improve the results of the programme.
        And ta-lib can be called. But this time we have programmed all the technical indicators to demonstrate the python techniques.
    '''
    #CCI
    df['CCI'] = tc.CCI(df)
    df['CCI_5D_mean'] = df['CCI'].rolling(5).mean()
    df['CCI_5D_std'] = df['CCI'].rolling(5).std()
    df['CCI_13D_mean'] = df['CCI'].rolling(13).mean()
    df['CCI_13D_std'] = df['CCI'].rolling(13).std()
    df['CCI_21D_mean'] = df['CCI'].rolling(21).mean()
    df['CCI_21D_std'] = df['CCI'].rolling(21).std()
    #MACD
    df['MACD'] = tc.MACD(df)
    df['MACD_signal'] = df['MACD'].ewm(span = 9, adjust = False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    #RSI
    df['RSI'] =tc.RSI(df)
    df['RSI_5D_mean'] = df['RSI'].rolling(5).mean()
    df['RSI_5D_std'] = df['RSI'].rolling(5).std()
    df['RSI_13D_mean'] = df['RSI'].rolling(13).mean()
    df['RSI_13D_std'] = df['RSI'].rolling(13).std()
    df['RSI_21D_mean'] = df['RSI'].rolling(21).mean()
    df['RSI_21D_std'] = df['RSI'].rolling(21).std()
    #WillR
    df['Will_R'] = tc.Will_R(df)
    #ADX
    df['ADX'] = tc.ADX(df)
    #Triple Exponential Moving Average
    df['TRIX'] = tc.TRIX(df)
    #On-balance Volume
    df['OBV'] = tc.OBV(df)
    df['OBV_chg'] = df['OBV'].pct_change()
    #PX_Slope
    df['PX_slope'] = tc.PX_slope(df)
    df['PX_slope_chg'] = df['PX_slope'].diff()

def standardize(df):
    #Standardize data for comparison
    df['s_High'] = df['High'] / df['Open']
    df['s_Low'] = df['Low'] / df['Open']
    df['s_Close'] = df['Close'] / df['Open']
    df['s_HH'] = df['HH'] / df['Close']
    df['s_LL'] = df['LL'] / df['Close']



