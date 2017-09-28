import pandas as pd
import numpy as np


# min (a*x+b -y)^2
def predict(a,b,c,x):
    return a*(x**2)+b*x+c

def loss(a,b,c,x,y):
    n = len(x)
    return np.sum((a*(x**2)+b*x+c-y)**2)/n + 0.0001*(a**2+b**2+c**2)

def gradient_a(a, b, c, x, y):
    n = len(x)
    temp = 2*np.sum((a*(x**2)+b*x+c-y)*(x**2))/n
    temp += 0.0001*2*a
    return temp

def gradient_b(a, b, c, x, y):
    n = len(x)
    temp = 2*np.sum((a*(x**2)+b*x+c-y)*(x))/n
    temp += 0.0001*2*b
    return temp

def gradient_c(a, b, c, x, y):
    n = len(x)
    temp = 2*np.sum(a*(x**2)+b*x+c-y)/n
    temp += 0.0001*2*c
    return temp



if __name__ == "__main__":
    df = 'D:/Myproject/stockPrice/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.zip'
    
    df = pd.read_csv(df)
    
    stock_ticker = df.groupby(df.ticker).sum().index.values

    result = np.empty((len(stock_ticker), ), dtype=[('ticker', 'S10'), ('predict_adj_open', 'f4')])
    index = 0
    for ticker in stock_ticker:
        stock = df.loc[df.ticker==ticker, ['adj_volume', 'adj_open', 'adj_high', 'adj_low', 'adj_close']]
        
        yopen = stock.values[:, 1]
        xopen = np.arange(*yopen.shape)
        result_lost = np.empty(len(xopen))

        a = 0
        b = 0
        c = 0
        lr = 0.0000000000000001 # Why this is bad? And how to improve it.
        for i in range(1):
            for j in range(len(xopen)):
                a -= lr*gradient_a(a, b, c, xopen, yopen)
                b -= lr*gradient_b(a, b, c, xopen, yopen)
                c -= lr*gradient_c(a, b, c, xopen, yopen)
                
                result_lost[j] = loss(a, b, c, xopen, yopen)
                if j>=1:
                    if result_lost[j]<result_lost[j-1]:
                        lr *= 1.05
                    else:
                        lr *= 0.5
                #print(a, b, xopen[j], yopen[j], loss(a, b, xopen, yopen))

        result[index]['ticker'] = ticker
        result[index]['predict_adj_open'] = predict(a,b,c,len(xopen))
#        print(loss(a,b,c,xopen,yopen))
        print(result[index])
        index += 1
    np.save('tomorrow_stock', result)
