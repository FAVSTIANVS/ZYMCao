import numpy as np 
from scipy.stats import norm
import scipy as sp
from functools import reduce
import math


# S: Underlying price, in home currency
# K: Strike price, in home currency
# r: Risk-free rate, unitless as a decimal
# T: Tenor, in years
# Y: Option price observed, in home currency

d1 = lambda S, K, r, T, sigma: (np.log(S/K) + (r + (sigma ** 2)/2) * T) / (sigma * math.sqrt(T)) 
# Helping function from the first assignment, "BS Implementation Using Python.py"
d2 = lambda S, K, r, T, sigma: d1(S, K, r, T, sigma) - sigma * math.sqrt(T)
# Helping function
C = lambda S, K, r, T, sigma: S * norm.cdf(d1(S, K, r, T, sigma)) - K * np.exp(-r * T) * norm.cdf(d2(S, K, r, T, sigma))
# Call price from the first assignment
P = lambda S, K, r, T, sigma: K * np.exp(-r * T) * norm.cdf(-d2(S, K, r, T, sigma)) - S * norm.cdf(-d1(S, K, r, T, sigma))
# Put price from the first assignment
vega = lambda S, K, r, T, sigma: S * norm.pdf(d1(S, K, r, T, sigma)) * math.sqrt(T)


def biC(S, K, r, T, Y, guess = 2000):
# Using bisection method to calculate the implied volativity, imputs are NOT vectorized
    vol1 = np.finfo(float).eps # numpy.finfo(float).eps gives a tiny positive number 
    vol2 = guess 
    vol3 = (vol1 + vol2)/2
    C1 = C(S, K, r, T, vol1)
    C2 = C(S, K, r, T, vol2)
    C3 = C(S, K, r, T, vol3)
    it = 0
    if   np.abs(C1 - Y) < np.finfo(float).eps:
        return [vol1, it]
    elif np.abs(C2 - Y) < np.finfo(float).eps:
        return [vol2, it]
    elif np.abs(C3 - Y) < np.finfo(float).eps:
        return [vol3, it]
    else:
        while np.abs(C3 - Y) > np.finfo(float).eps:
            C1 = C(S, K, r, T, vol1)
            C2 = C(S, K, r, T, vol2)
            if (C1-Y) * (C3-Y) < 0:
                vol1 = vol1
                vol2 = vol3
                vol3 = (vol1 + vol2)/2
                C3 = C(S, K, r, T, vol3)
                it = it + 1
            elif (C2-Y) * (C3-Y) < 0 or np.abs(C2-Y) < np.abs(C1-Y):
                vol1 = vol3
                vol2 = vol2
                vol3 = (vol1 + vol2)/2
                C3 = C(S, K, r, T, vol3)
                it = it + 1
            else:
                vol1 = vol1
                vol2 = vol3
                vol3 = (vol1 + vol2)/2
                C3 = C(S, K, r, T, vol3)
                it = it + 1
        return [vol3, it]

# biC(50, 50, 0.05, 1, 10.89)

# fmin test
# print(sp.optimize.fmin(func=lambda x: np.abs(C(50, 50, 0.05, 1, x) - 10.89), x0=[0.15]))



def BIC(SS, KK, rr, TT, YY):
# Using bisection method to calculate implied volativities, imputs are INDEED vectorized
# Inputs are all "vectorized" lists 
    Output = [None] * len(YY)
    for i in range(len(YY)):
        Output[i] = biC(SS[i], KK[i], rr[i], TT[i], YY[i])
    Z = np.array(Output)
    print("the vector of implied volativities are:" , Z[:, 0])
    print("the vector of iterations taken are:", Z[:, 1], "respectively")


def newtC(S, K, r, T, Y, guess = 1.6):
# Using Newton's method to calculate the implied volativity, imputs are NOT vectorized
    vol = guess
    it = 0
    while np.abs(C(S, K, r, T, vol) - Y) > np.finfo(float).eps and it < 10 :  #np.finfo(float).eps:
        vol = (Y - C(S, K, r, T, vol)) / vega(S, K, r, T, vol) + vol
        it = it + 1
    return [vol, it]


def NEWTC(SS, KK, rr, TT, YY):
# Using Newton's method to calculate implied volativities, imputs are INDEED vectorized
# Inputs are all "vectorized" lists 
    Output = [None] * len(YY)
    for i in range(len(YY)):
        Output[i] = newtC(SS[i], KK[i], rr[i], TT[i], YY[i])
    Z = np.array(Output)
    print("the vector of implied volativities are:" , Z[:, 0])
    print("the vector of iterations taken are:", Z[:, 1], "respectively")

# Test
SS = [50] * 100 
KK = [50] * 100
rr = [0.05] * 100
TT = [1] *100
YY = [10.89] * 100

NEWTC(SS, KK, rr, TT, YY)
BIC(SS, KK, rr, TT, YY)