import os
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from subprocess import check_call
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

'''
This script tries to detect momentum effect of stocks using all its necessary features.

RandomForest is used as the classifier.
We train our model by different holding period as target variable, which means we are trying to predict the n-day return based 
    current information.

Levels of the tree is limited to 5 to prevent overfitting.

'''

main_data_path = os.getcwd() + '\\main_data\\HONGKONG_5Y.csv'
backtest_path = os.getcwd() + '\\main_data\\HONGKONG_backtest.csv'

os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

if __name__ == '__main__':
    df = pd.read_csv(main_data_path) #Read the data
    df.set_index(['Date', 'Ticker'], inplace=True)
    #Import the model
    model = RandomForestClassifier(n_estimators=60, random_state=0, max_depth=5)
    #Set up training parameters
    best = 0
    precision_score = np.empty([15])
    for n in range(1, 15):
        train = df.copy(deep=True)
        train['Holding'] = train.groupby('Ticker')['Close'].apply(lambda x: np.log(x / x.shift(n)))
        train['Target'] = train.groupby('Ticker')['Holding'].shift(-n)
        train['Target'] = train['Target'].combine(train['Holding'],
                                                  lambda x1, x2: 1 if (x1 > 0 and x2 > 0) or (x1 < 0 and x2 < 0) else 0)
        train = train.astype({'Volume_rank': 'float32', 'High_rank': 'float32'})
        train = train.replace([np.inf, -np.inf], np.nan)
        train = train.dropna()
        #Get features and target variable
        x = train.drop(['Target', 'Open', 'High', 'Low', 'Close',
                        'HH', 'LL', 'Adj Close', 'Volume', 'OBV', 'Holding'], axis=1)
        y = train['Target']
        #70/30 is used as the ratio for training and validation dataset
        train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.3,
                                                                                    random_state=30)
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        # Result
        result = predictions - test_labels
        accuracy = 1 - abs(result).sum() / len(predictions)
        tp = (predictions[predictions == 1].sum() - (result[result == 1].sum()))
        precision = tp / predictions[predictions == 1].sum()
        print(
            f'Holding: {n}, Accuracy: {accuracy}, Precision: {precision}, Train Rows: {len(train_features)}, Test Rows: {len(test_features)}')
        precision_score[n] = precision
        if precision > best:
            best_n = n
            best = accuracy
            best_model = model

    #Plot one of the tree in the best random forest
    estimator =best_model.estimators_[5]
    export_graphviz(estimator, out_file='tree.dot', max_depth=10,
                    feature_names=x.columns,
                    filled=True, rounded=True,
                    special_characters=True)
    check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi = 600'])
    #Plot fitting result
    plt.figure(figsize = (10, 5))
    plt.plot(precision_score[1:])
    plt.grid()
    plt.title('Accuracy with increasing complexity')
    plt.show()
