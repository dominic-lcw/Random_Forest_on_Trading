import os
import glob
import pandas as pd

from feature import fundamental
from feature import technical
from feature import clean
from feature import standardize

'''
This script process all files on the 'source_path', calculate different features as specified on "feature.py" and 
save them to 'target_path'.
'''

source_path = os.getcwd() + '\\source\\*.csv'
target_path = os.getcwd() + '\\individual'

if __name__ =='__main__':
    for fname in glob.glob(source_path):
        df = pd.read_csv(fname)
        clean(df)
        fundamental(df)
        technical(df)
        standardize(df)
        df.drop(['Date'], axis = 1, inplace = True)
        save = target_path + "\\" + os.path.split(fname)[1]
        df.to_csv(save)