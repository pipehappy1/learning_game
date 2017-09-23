import pandas as pd
import numpy as np


# min (a*x+b -y)^2
def predict(a,b,x):
    return a*x+b

def loss(a,b,x,y):
    n = len(x)
    return np.sum((a*x +b -y)**2)/n

def gradient_a(a, b, x,y):
    return 2*(a*x+b-y)*x

def gradient_b(a, b, x, y):
    return 2*(a*x+b-y)

# preprocessing
def pp(x, y, tx):
    xstd = np.std(x)
    xmean = np.mean(x)

    ystd = np.std(y)
    ymean = np.mean(y)

    return (x-xmean)/xstd, (y-ymean)/ystd, (tx-xmean)/xstd, ymean, ystd


if __name__ == "__main__":
    df = '/hdd/home/largedata/training_game/us_stock_2017_09_11.zip'
    
    df = pd.read_csv(df)
    
    stock_ticker = df.groupby(df.ticker).sum().index.values

    result = np.empty((len(stock_ticker), ), dtype=[('ticker', 'S10'), ('predict_adj_open', 'f4')])
    index = 0
    for ticker in stock_ticker:
        stock = df.loc[df.ticker==ticker, ['adj_volume', 'adj_open', 'adj_high', 'adj_low', 'adj_close']]
        
        yopen = stock.values[:, 1]
        xopen = np.arange(*yopen.shape)

        x, y, tx, ymean, ystd = pp(xopen, yopen, len(xopen)+1)
        
        a = 0.1
        b = 0
        lr = 0.01 # Why this is bad? And how to improve it.
        for i in range(10):
            for j in range(len(xopen)):
                a -= lr*gradient_a(a, b, x[j], y[j])
                b -= lr*gradient_b(a, b, x[j], y[j])
                #print(a, b, xopen[j], yopen[j], loss(a, b, xopen, yopen))

        result[index]['ticker'] = ticker
        result[index]['predict_adj_open'] = predict(a,b,tx)*ystd+ymean
        print(result[index]['predict_adj_open'])
        index += 1
    np.save('tomorrow_stock', result)

    
    
    
