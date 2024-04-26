import numpy as np
import random as random

def linear_regression(x, y, iteration=100, lr=0.01):
    n, m = len(x[0]), len(x)
    b0, bother=initialize_param(n)
    for _ in range(iteration):
        grb0,grbother=compute_gr(x,y, b0, bother, n,m)
        b0, bother=update_param(b0,bother, grb0, grbother, lr)
    return b0, bother

def initialize_param(dimension):
    b0=0
    bother=[random.random() for _ in range(dimension)]
    return b0, bother

def compute_gr(x,y,b0,bother,n,m):
    grb0=0
    grbother=[0]*n
    for i in range(m):
        yhat=sum(bother[j]*x[i][j] for j in range(n))+b0
        derr= 2*(y[i]-yhat)
        for j in range(n):
            grbother[j]+=derr*x[i][j]/n
        grb0+=derr/n
    return grb0, grbother

def update_param(b0,bother,grb0, grbother, lr):
    b0+=grb0*lr
    for i in range(len(grbother)):
        bother[i]+=grbother[i]*lr
    return b0,bother

np.random.seed(0)
X = np.random.rand(100, 5)  # Example: 100 samples, 2 features
y = np.random.randint(0, 2, size=100)  # Example: Binary labels

# 2. Call logistic regression function
beta_0, beta_other = linear_regression(X, y, 100, 0.01)

# 3. Inspect the results
print("Beta_0:", beta_0)
print("Beta_other:", beta_other)