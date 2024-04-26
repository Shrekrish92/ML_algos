import numpy as np
import random as random

def logistic_regression(x,y,iteration=100, learning_rate=0.01):
    m,n=len(x),len(x[0])
    beta_0, beta_other= initialise_param(n)
    for _ in range(iteration):
        gradient_beta_0, gradient_beta_other=compute_gradient(x,y,beta_0, beta_other,m,n)
        beta_0, beta_other=update_param(beta_0, beta_other, gradient_beta_0,gradient_beta_other, learning_rate)
    return beta_0, beta_other

def initialise_param(n):
    beta_0=0
    beta_other=[random.random() for _ in range(n)]
    return beta_0, beta_other

def compute_gradient(x, y, beta_0, beta_other, m, n):
    gradient_beta_0=0
    gradient_beta_other=[0]*n
    for i, point in enumerate(x):
        pred=logistic_function(point, beta_0, beta_other)
        for j, feature in enumerate(point):
            gradient_beta_other[j]+=(pred-y[i])*feature/m
            gradient_beta_0+=(pred-y[i])/m
    return gradient_beta_0,gradient_beta_other

def logistic_function(point, beta_0, beta_other):
    return 1/(1 + np.exp(-(beta_0 + point.dot(beta_other))))

def update_param(beta_0, beta_other, gradient_beta_0, gradient_beta_other, learning_rate):
    beta_0 -= gradient_beta_0 * learning_rate
    for i in range(len(beta_other)):
        beta_other[i] -= (gradient_beta_other[i] * learning_rate)
    return beta_0, beta_other

np.random.seed(0)
X = np.random.rand(100, 2)  # Example: 100 samples, 2 features
y = np.random.randint(0, 2, size=100)  # Example: Binary labels

# 2. Call logistic regression function
beta_0, beta_other = logistic_regression(X, y, iteration=1000, learning_rate=0.01)

# 3. Inspect the results
print("Beta_0:", beta_0)
print("Beta_other:", beta_other)