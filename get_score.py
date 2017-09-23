import pandas as pd
import numpy as np
import math
import os
import pickle

if __name__ == "__main__":
    predict_912 = np.load('./tomorrow_stock.npy')
    cleaned = {}
    for i in predict_912:
        val = i['predict_adj_open']
        if np.isnan(val):
            val = -1000000
        if np.isinf(val):
            val = -1000000
        cleaned[i['ticker'].decode()] = val

    true_price = {}
    true_price_pickle = './us_stock_2017_09_12.pickle'
    if not os.path.isfile(true_price_pickle):
        real_stock_price = '/hdd/home/largedata/training_game/us_stock_2017_09_14.zip'
        df = pd.read_csv(real_stock_price)
        for i in df.loc[df.date=='2017-09-12', ('ticker', 'adj_open')].values:
            true_price[i[0]] = i[1]
        with open(true_price_pickle, 'wb') as df:
            pickle.dump(true_price, df)
    else:
        with open(true_price_pickle, 'rb') as df:
            true_price = pickle.load(df)

    score = 0
    for k, v in true_price.items():
        if cleaned[k] is not None:
            score += (cleaned[k]-v)**2
    score = math.sqrt(score)
    print(score)

    
        
    
    

    