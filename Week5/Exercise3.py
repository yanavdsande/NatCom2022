
# %%
#import statements
import math
import matplotlib.pyplot as plt 
import numpy as np 

#%%
def comb(n,r):
    """ A function to calculate the combination operator, returns the number of combations that is possible when selecting r out of n"""
    return (math.factorial(n))/ (math.factorial(n-r) * math.factorial(r))
#%%
def strong_classifier_right(r, n):
    """ Calculates the possibility that a correct majority vote is reached considering the strong classifier is right  """
    return comb(n,r) * 0.6**r * 0.4**(n-r)  * 0.8

def strong_classifier_wrong(r, n):
    """Calculates the possibility that a correct majority vote is reached considering the strong classifier is wrong"""
    return comb(n,r) * 0.6**r * 0.4 ** (n-r) * 0.2

#%%
def weighted_ensemble(weights, n):
    prob_with_sc = 0 
    prob_without_sc = 0
    weighted_prob = []

    for w in weights:
        votes = 1*n + w*1  #aantal weak classifiers #
        majority = votes/2. # Everything above is a majority
        for r in range(0, n+1):
            if r+w > majority: #case that the strong classifier is right
                print(r)
                prob_with_sc += strong_classifier_right(r, n)
            else: prob_with_sc += 0 
            if r > majority: #case that the strong classifier is wrong
                prob_without_sc += strong_classifier_wrong(r, n)
            else: prob_without_sc += 0
        weighted_prob.append(prob_without_sc+prob_with_sc) #counts the probability for a correct majority vote per weight
        prob_with_sc = 0  #resets the counter
        prob_without_sc = 0 #resets the counter 
    
    #plots the weighted probability
    plt.figure()
    plt.plot(weights, weighted_prob)
    plt.title('Weighted probability')
    plt.xlabel('Weight given to the strong classifier')
    plt.ylabel('Probability of a correct majority vote')
    plt.show()
    return weighted_prob

# %% 
#runs the code with different weights 
weighted_ensemble([1,2.6,3,4,5,6,7,8, 9, 10, 11, 12, 13], 10)
# %%
#This is exercise c: 
#%%
def error(n, weights):
    """ weight calculation in the adaboost algorithm
        n: amount of classifiers 
        weights: initial weight
        returns: err_m"""
    denom = 0
    som = 0
    for i in range(0,n):
        if i < n-1: 
            product = weights[i] * 0.6
        else: 
            product = weights[i] * 0.2
        denom += weights[i]
        som += product

    error = som / denom 
    return error

#%%
def alpha(error):
    """ Calculates the alpha value in the adaboost algorithm. 
    error: err_m 
    returns: the alpha value """
    return np.log((1-error)/error)

#%%
def weight_update(n, weights, alpha):
    """ Weight update in adaboost algorithm. 
        n = number of classifier
        weights = the previous weights 
        alpha = alpha value 
        returns: updated weights
    """
    for i in range(0,n):
        if i < n-1: 
            weights[i] = weights[i] * np.exp(alpha * 0.9)
        else: 
            weights[i] = weights[i] * np.exp(alpha * 0.2)
    return weights
#%%
def adaboost(M,n): 
    """implementation of the adaboost algorithm, 
    M: interations 
    n: number of classifiers 
    returns: updated weights after M iterations, the weight_errors, and the weights for the first classifier in the ensemble """
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
#runs the adaboost algorithm and prints the last weights
weights, err, wei = adaboost(10, 11)
print(weights)
# %%
#tracks the weight update of the first classifier in the ensemble and plots the update
print(wei)
plt.figure()
plt.plot(err, wei)
plt.title('AdaBoost weight per error under a misclassification probability of 0.4')
plt.xlabel('Error')
plt.ylabel('weight')
   

# %%
#calculates the baselearner weight and plots the change

base_learner = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpharate= []
for i in base_learner:
    alpharate.append(alpha(i))

plt.figure
plt.title("Alpha value per error-rate of a baselearner")
plt.plot(base_learner, alpharate)
plt.xlabel("Error rate baselearner")
plt.ylabel("Alpha value")
plt.show()



# %%
