import pandas as pd
import numpy as np


# min (a*x+b -y)^2
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
    stock_A = df.loc[df.ticker=='A', ['adj_volume', 'adj_open', 'adj_high', 'adj_low', 'adj_close']]

    yopen = stock_A.values[:, 1]
    xopen = np.arange(*yopen.shape)

    a = 0.1
    b = 0
    lr = 0.00000001 # Why this is bad? And how to improve it.
    for i in range(10):
        print('epoch: {}'.format(i))
        for j in range(len(xopen)):
            a -= lr*gradient_a(a, b, xopen[j], yopen[j])
            b -= lr*gradient_b(a, b, xopen[j], yopen[j])
            print(a, b, xopen[j], yopen[j], loss(a, b, xopen, yopen))

    
    
    
