import numpy as np 
from scipy.stats import norm
import math
from numpy import matlib

# S: Underlying price, in home currency
# K: Strike price, in home currencyu
# r: Risk-free rate, unitless as a decimal
# T: Tenor, in years
# sigma: Volatility, unitless as a decimal
# N: number of steps
# M: the Mth step, only in use in auxiliary function
# q: dividend

u = lambda sigma, T, N: np.exp( sigma * math.sqrt(T/N))
# price up multiplier per tick

d = lambda sigma, T, N: np.exp(-sigma * math.sqrt(T/N))
# price down multiplier per tick

p = lambda r, T, sigma, N, q: ( np.exp( (r-q)*(T/N) ) - d(sigma, T, N) )/(u(sigma, T, N) - d(sigma, T, N))
# auxiliary variable p

def pair(z):
# this is a bijection between natural numbers and its cartesian product N*N; I was trying to save data in a list instead of a matrix, 
# but time is not enough, so this function is not really in use
    w = math.floor((math.sqrt(8*z + 1) - 1)/2)
    t = (w**2 + w)/2
    y = z - t
    x = w - y
    A = [x, y]
    return A

def V(S, T, sigma, N, M):
# this calculates the possible underlying prices at the Mth step when the overall steps are N
    Output = [None] * (M+1)
    for i in range(M+1):
        Output[i] = S * pow(d(sigma, T, N), i)  * pow(u(sigma, T, N), M - i)
    Output = np.array(Output)
    return Output

# V(100, 1, 0.2, 3, 3)

CEnd = lambda S, K, T, sigma, N: np.array(list(map(lambda x: max(x, 0), V(S, T, sigma, N, N) - K)))
# calculate the call prices at the end of T

PEnd = lambda S, K, T, sigma, N: np.array(list(map(lambda x: max(x, 0), -V(S, T, sigma, N, N) + K)))
# calculate the put prices at the end of T


def ECprev(S, K, r, T, sigma, N, q, vector):
# helping function
# given vector, which is the the set of call prices at a particular tick, calculate the European call prices at the previous tick
    t = T/N
    p = ( np.exp( (r-q)*(T/N) ) - d(sigma, T, N) )/(u(sigma, T, N) - d(sigma, T, N))
    Output = [None] * (len(vector) - 1)
    for i in range(len(Output)):
        Output[i] = np.exp(- r * t) * (p * (vector[i]) + (1-p) *vector[i + 1] )
    return Output


def EC(S, K, r, T, sigma, N, q):
# calculate the European call price at the beginning of time
    Output = CEnd(S,K,T,sigma,N)
    for i in range(N):
        Output = ECprev(S,K,r,T,sigma,N,q,Output)
    return Output[0]

def EP(S, K, r, T, sigma, N, q):
# calculate the European put price at the beginning of time
    Output = PEnd(S,K,T,sigma,N)
    for i in range(N):
        Output = ECprev(S,K,r,T,sigma,N,q,Output)
    return Output[0]

# EC(50,50,0.05,1,0.5,1000,0)
# EP(50,50,0.05,1,0.5,1000,0)

def ACprev(S, K, r, T, sigma, N, q, vector):
# given vector, which is the the set of call prices at a particular tick, calculate the American call prices at the previous tick
    t = T/N
    p = ( np.exp( (r-q)*(T/N) ) - d(sigma, T, N) )/(u(sigma, T, N) - d(sigma, T, N))
    Output = [None] * (len(vector) - 1)
    UPx = V(S, T, sigma, N, len(vector) - 2) # the underlying prices at the previous tick
    for i in range(len(Output)):
        Output[i] = max(np.exp(- r * t) * (p * (vector[i]) + (1-p) *vector[i + 1] ), UPx[i] - K)
    return Output

def APprev(S, K, r, T, sigma, N, q, vector):
# helping function
# given vector, which is the the set of put prices at a particular tick, calculate the American put prices at the previous tick
    t = T/N
    p = ( np.exp( (r-q)*(T/N) ) - d(sigma, T, N) )/(u(sigma, T, N) - d(sigma, T, N))
    Output = [None] * (len(vector) - 1)
    UPx = V(S, T, sigma, N, len(vector) - 2) # the underlying prices at the previous tick
    for i in range(len(Output)):
        Output[i] = max(np.exp(- r * t) * (p * (vector[i]) + (1-p) *vector[i + 1] ), -UPx[i] + K)
    return Output


def AC(S, K, r, T, sigma, N, q):
# calculate the American call price at the beginning of time
    Output = CEnd(S,K,T,sigma,N)
    for i in range(N):
        Output = ACprev(S,K,r,T,sigma,N,q,Output)
    return Output[0]


def AP(S, K, r, T, sigma, N, q):
# calculate the American put price at the beginning of time
    Output = PEnd(S,K,T,sigma,N)
    for i in range(N):
        Output = APprev(S,K,r,T,sigma,N,q,Output)
    return Output[0]

# AC(50,50,0.05,1,0.5,1000,0)
# AP(50,50,0.05,1,0.5,1000,0)

def Delta(S, K, r, T, sigma, N, q):
    A = CEnd(S,K,T,sigma,N) # call prices when t = T
    for i in range(N - 1):
        A = APprev(S,K,r,T,sigma,N,q,A) # call prices when t = 1
    B = V(S, T, sigma, N, 1) # underlying prices when t = 1
    dC = A[0] - A[1]
    dS = B[0] - B[1]
    delta = (dC) / (dS)
    return delta

def Gamma(S, K, r, T, sigma, N, q):
    A = CEnd(S,K,T,sigma,N) # call prices when t = T
    for i in range(N - 2):
        A = APprev(S,K,r,T,sigma,N,q,A) # call prices when t = 2
    B = V(S, T, sigma, N, 2) # underlying prices when t = 2
    gamma = ( (A[0] - A[1])/(B[0] - B[1]) - (A[1] - A[2])/(B[1] - B[2]) )/((B[0] - B[2])/2)
    return gamma

Vega = lambda S, K, r, T, sigma, N, q: (EC(S, K, r, T, sigma * 1.01, N, q) - EC(S, K, r, T, sigma * 0.99, N, q))/(2* 0.01 * sigma)

def Theta(S, K, r, T, sigma, N, q):
    A = CEnd(S,K,T,sigma,N) # call prices when t = T
    t = T/N
    for i in range(N - 2):
        A = APprev(S,K,r,T,sigma,N,q,A) # call prices when t = 2
    theta = A[0] - EC(S, K, r, T, sigma, N, q)/(2*t)
    return theta

Rho = lambda S, K, r, T, sigma, N, q: (EC(S, K, r*1.01, T, sigma, N, q) - EC(S, K, r*0.99, T, sigma, N, q))/(2* 0.01 * r)

# test results:
# print("European call price: ", EC(50, 50,0.05,1,0.5,1000,0))
# EP(50, 50,0.05,1,0.5,1000,0)
# Delta(50, 50,0.05,1,0.5,1000,0)
# Gamma(50, 50,0.05,1,0.5,1000,0)
# Vega(50, 50,0.05,1,0.5,1000,0)
# Theta(50, 50,0.05,1,0.5,1000,0)
# Rho(50, 50,0.05,1,0.5,1000,0)