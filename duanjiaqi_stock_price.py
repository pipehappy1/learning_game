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


if __name__ == "__main__":
    df = '/hdd/home/largedata/training_game/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.zip'
    
    df = pd.read_csv(df)
    
    stock_ticker = df.groupby(df.ticker).sum().index.values

    result = np.empty((len(stock_ticker), 1))
    index = 0
    for ticker in stock_ticker:
        stock = df.loc[df.ticker==ticker, ['adj_volume', 'adj_open', 'adj_high', 'adj_low', 'adj_close']]
        
        yopen = stock.values[:, 1]
        xopen = np.arange(*yopen.shape)
        
        a = 0.1
        b = 0
        lr = 0.00000001 # Why this is bad? And how to improve it.
        for i in range(1):
            for j in range(len(xopen)):
                a -= lr*gradient_a(a, b, xopen[j], yopen[j])
                b -= lr*gradient_b(a, b, xopen[j], yopen[j])
                
                if j%5==0:
                    lr *= 0.99 # Learning-rate decreases with the iteration.
                    
                #print(a, b, xopen[j], yopen[j], loss(a, b, xopen, yopen))

        result[index,0] = predict(a,b,len(xopen)+1)
        index += 1
    np.save('tomorrow_stock', result)

    
    
    
