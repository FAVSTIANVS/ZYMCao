import numpy as np 
from scipy.stats import norm
import math

# S: Underlying price, in home currency
# K: Strike price, in home currency
# r: Risk-free rate, unitless as a decimal
# T: Tenor, in years
# sigma: Volatility, unitless as a decimal

C = lambda S, K, r, T, sigma: S * norm.cdf(d1(S, K, r, T, sigma)) - K * np.exp(-r * T) * norm.cdf(d2(S, K, r, T, sigma))
# Call price
P = lambda S, K, r, T, sigma: K * np.exp(-r * T) * norm.cdf(-d2(S, K, r, T, sigma)) - S * norm.cdf(-d1(S, K, r, T, sigma))
# Put price
d1 = lambda S, K, r, T, sigma: (math.log(S/K) + (r + pow(sigma, 2)/2) * T) / (sigma * math.sqrt(T))
d2 = lambda S, K, r, T, sigma: d1(S, K, r, T, sigma) - sigma * math.sqrt(T)
deltaC = lambda S, K, r, T, sigma: norm.cdf(d1(S, K, r, T, sigma))
deltaP = lambda S, K, r, T, sigma: norm.cdf(d1(S, K, r, T, sigma)) - 1
gamma = lambda S, K, r, T, sigma: norm.pdf(d1(S, K, r, T, sigma)) / (S * sigma * math.sqrt(T))
vega = lambda S, K, r, T, sigma: S * norm.pdf(d1(S, K, r, T, sigma)) * math.sqrt(T)
thetaC = lambda S, K, r, T, sigma: - (S * norm.pdf(d1(S, K, r, T, sigma)) * sigma)/(2 * math.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2(S, K, r, T, sigma))
thetaP = lambda S, K, r, T, sigma: - (S * norm.pdf(d1(S, K, r, T, sigma)) * sigma)/(2 * math.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(d2(S, K, r, T, sigma))
rhoC = lambda S, K, r, T, sigma: K * T *np.exp(-r * T) * norm.cdf(d2(S, K, r, T, sigma))
rhoP = lambda S, K, r, T, sigma: - K * T *np.exp(-r * T) * norm.cdf(-d2(S, K, r, T, sigma))

# for i in range(10):
#     names = ["C: ", "P: ", "deltaC: ", "deltaP: ", "gamma: ", "vega: ", "thetaC: ", "thetaP: ", "rhoC: ", "rhoP: "]
#     f = [C, P, deltaC, deltaP, gamma, vega, thetaC, thetaP, rhoC, rhoP]
#     print(names[i], f[i](50, 50, 0.05, 1, 0.5))