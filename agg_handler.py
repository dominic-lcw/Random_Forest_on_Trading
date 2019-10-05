import os
import glob
import pandas as pd

'''
This Script generates dataset to train the model (both training and testing) and the holdup dataset (most recent 30days) for backtestging

Operation:
Aggreagate all csv and produce training data to 'train_data_path' and backtest data to 'backtest_path'
'''


source_path = os.getcwd() + '\\individual\\*.csv'
train_data_path = os.getcwd() + '\\main_data\\HONGKONG_5Y.csv'
backtest_path = os.getcwd() + '\\main_data\\HONGKONG_holdup.csv'

if __name__ == '__main__':
    new = True
    for fname in glob.glob(source_path):
        stock = pd.read_csv(fname)
        name = os.path.split(fname)[1][:-4]
        stock['Ticker'] = name
        if new:
            df = pd.DataFrame(columns = stock.columns)
            backtest = pd.DataFrame(columns = stock.columns)
            new = False
        df = df.append(stock[:-30], sort = False)
        backtest = backtest.append(stock[-30:], sort = False)
    #Get Cross_sectional Rank
    df.set_index(['Date', 'Ticker'], inplace = True)
    df['Volume_rank'] = df.groupby('Date')['Volume'].apply(lambda x: x.rank())
    df['High_rank'] = df.groupby('Date')['High'].apply(lambda x: x.rank())
    backtest.set_index(['Date', 'Ticker'], inplace=True)
    backtest['Volume_rank'] = backtest.groupby('Date')['Volume'].apply(lambda x: x.rank())
    backtest['High_rank'] = backtest.groupby('Date')['High'].apply(lambda x: x.rank())
    #Save the dataset
    df.to_csv(train_data_path)
    backtest.to_csv(backtest_path)


    # #Get Cross_sectional feature
    # alpha4 = df.groupby('Ticker')['Volume_rank'].rolling(9).apply(lambda x: np.argsort(x)[-1], raw = True)
    # alpha4.index = alpha4.index.droplevel()
    # df['Alpha#4'] = alpha4
    #Add: more cross-sectional alpha features



