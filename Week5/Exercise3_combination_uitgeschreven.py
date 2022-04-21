
# %%
import math
from msilib.schema import Error
import numpy as np
import scify 
import matplotlib.pyplot as plt 
#%%
def comb(n,r):
    return np.math.factorial(n)/ (np.math.factorial(n-r) * np.factorial(r))
#%%
def strong_classifier_right(r, n):
    return comb(n,r) * 0.6**r * 0.4**(n-r)  * 0.8

def strong_classifier_wrong(r, n):
    return comb(n,r) * 0.6**r * 0.4 ** (n-r) * 0.2

#%%
def weighted_ensemble(weights, n):
    prob_with_sc = 0 
    prob_without_sc = 0
    weighted_prob = []


    for w in weights:
        votes = 1*n + w*1 
        majority = votes/2 
        for r in range(0, n+1):
            if r+w > majority: 
                prob_with_sc += strong_classifier_right(r, n)
            else: prob_with_sc += 0
            if r>majority:
                prob_without_sc += strong_classifier_wrong(r, n)
            else: prob_without_sc += 0
        weighted_prob.append(prob_without_sc+prob_with_sc)
        prob_with_sc = 0 
        prob_without_sc = 0

    
    plt.figure()
    plt.plot(weights, weighted_prob)
    plt.show()
    return weighted_prob




# %%
weighted_ensemble([1,2,3,4], 10)


# %%
#This is exercise 3: 
#%%
def error(n, weights):
    denom = 0
    som = 0
    for i in range(0,n):
        if i < n-1: 
            product = weights[i] * 0.4
        else: 
            product = weights[i] * 0.2
        denom += weights[i]
        som += product

    error = som / denom 
    return error


def alpha(error):
    return np.log((1-error)/error)

def weight_update(n, weights, alpha):
    for i in range(0,n):
        if i < n-1: 
            weights[i] = weights[i] * np.exp(alpha * 0.4)
        else: 
            weights[i] = weights[i] * np.exp(alpha * 0.2)
    return weights

def adaboost(M,n): 
    weights = np.full(n, 1/n)
    wei = []
    err = []
    for m in range(0, M):
        err.append(error(n, weights))
        weight_update(n, weights, alpha(error(n, weights)))
        wei.append(weights[0])
        # print(err)
        # print(wei)
    print(wei)
    
    return weights, err, wei
        
#%%
weights, err, wei = adaboost(10, 11)
print(weights)
# %%
print(wei)
plt.figure()
plt.plot(err, wei)
plt.title('AdaBoost weight per error under a misclassification probability of 0.4')
plt.xlabel('Error')
plt.ylabel('weight')

    

# %%
