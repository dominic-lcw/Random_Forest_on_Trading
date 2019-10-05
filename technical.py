import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def CCI(df, n = 20):
    '''Commodity Channel Index
    '''
    tp = (df['High'] + df['Low'] + df['Close']) /3
    cci = (tp - tp.rolling(n).mean()) / (tp.rolling(n).std() * 0.015)
    return cci

def MACD(df):
    '''Moving Average Convergence / Divergence
    '''
    ema12 = df['Close'].ewm(span = 12, adjust = False).mean()
    ema26 = df['Close'].ewm(span = 26, adjust = False).mean()
    macd = ema12 - ema26
    return macd

def RSI(df, n = 14):
    '''Relative Strength Index
    '''
    chg = df['Close'].diff()
    avg_gain = chg.rolling(n).apply(lambda x: x[x>0].sum() / n, raw = False)
    avg_loss = chg.rolling(n).apply(lambda x: -x[x<0].sum() / n, raw = False)
    rsi = 100 - (100 / (1+avg_gain / avg_loss))
    return rsi

def Will_R(df, n = 14):
    '''William % R
    '''
    highest_high = df['High'].rolling(n).max()
    lowest_low = df['Low'].rolling(n).min()
    will_r = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
    return will_r

def ADX(df, n = 14):
    '''Average Directional Index
    '''
    plus_dm = df['High'] - df['High'].shift(1)
    minus_dm = df['Low'].shift(1) - df['Low']
    plus_dm.dropna(inplace = True)
    minus_dm.dropna(inplace = True)
    mod_plus = plus_dm.combine(minus_dm, lambda x1, x2: x1 if x1 > x2 else 0)
    mod_minus = minus_dm.combine(plus_dm, lambda x1, x2: x1 if x1 > x2 else 0)
    #Calculate true range
    t1 = df['High'] - df['Low']
    t2 = df['High'] - df['Close'].shift(1)
    t3 = df['Low'] - df['Close'].shift(1)
    tr = t1.combine(t2, lambda x1, x2 : x1 if x1 > x2 else x2)
    tr = tr.combine(t3, lambda x1, x2 : x1 if x1 > x2 else x2)
    # Smoothed parameter
    mod_plus *= n
    mod_minus *= n
    tr *= n
    smooth_plus = mod_plus.ewm(alpha = 1/n, adjust = False).mean()
    smooth_minus = mod_minus.ewm(alpha = 1/n, adjust = False).mean()
    smooth_tr = tr.ewm(alpha = 1/n, adjust = False).mean()
    #Calculate ADX
    plus_d = (smooth_plus / smooth_tr) * 100
    minus_d = (smooth_minus / smooth_tr) * 100
    plus_d.iloc[:14] = None
    minus_d.iloc[:14] = None
    dx = abs(plus_d - minus_d) / (plus_d + minus_d) * 100
    adx = dx.ewm(alpha = 1/n, adjust = False).mean()
    return adx

def TRIX(df, n = 15):
    '''Triple Exponentially Smoothed Average
    '''
    ema1 = df['Close'].ewm(span = n).mean()
    ema2 = ema1.ewm(span = n).mean()
    ema3 = ema2.ewm(span = n).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    return trix

def Perc_B(df, n = 20):
    '''Bollinger Bands %B
    '''
    mid = df['Close'].rolling(n).mean()
    upper = mid +df['Close'].rolling(n).std() * 2
    lower = mid - df['Close'].rolling(n).std() * 2
    perc_b = (df['Close'] - lower) / (upper - lower) * 100
    return perc_b

def OBV(df):
    '''On-balance Volume
    '''
    diff = df['Close'].diff()
    plus_vol = df['Volume'].combine(diff, lambda x1, x2: x1 if x2 > 0 else 0)
    minus_vol = df['Volume'].combine(diff, lambda x1, x2: -x1 if x2 <0 else 0)
    obv_value = plus_vol + minus_vol
    return obv_value.cumsum()

def PX_slope(df, n = 14):
    '''Slope of the price series
    '''
    slope = np.zeros_like(df['Close'])
    y = pd.DataFrame(df['Close'])
    x = pd.DataFrame(y.reset_index().index.values, index = y.index.values)
    for i in range(n, len(y) +1):
        model = LinearRegression().fit(x.iloc[i-n:i, :], y.iloc[i-n:i, :])
        slope[i-1] = model.coef_
    slope = pd.Series(slope, index = df['Close'].index.values)
    return slope