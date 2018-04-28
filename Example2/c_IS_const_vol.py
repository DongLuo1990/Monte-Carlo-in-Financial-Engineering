# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:12:23 2017

@author: luodong
"""

### importance sampling for const volatility 
import time
import numpy as np
from scipy.optimize import fsolve
start = time.clock()
r = 0.05 # risk neutral interest rate
S0 = 50   # initial underlying asset price
T = 1  # maturity of the option
N = 32
K = 50 # strike price
V0 = 0.3*0.3 # initial volatility factor
rho = 0.5 # correlation between two brownian motions
Num = 1000000 # number of paths to simulate
dt = T/N #
rdt = np.sqrt(dt)
sigma = np.sqrt(V0)
drift = (r - 0.5*sigma**2)*dt
vol = sigma*rdt
discount = np.exp(-r*T)
# function for the formula 1/m*sum(s(tj,y)) - k -y =0
def func(y):
	S = np.zeros(N+1) # store the price
	Z = np.zeros(N+1) # 
	S[0] = S0 # initial price
	Z[1] = sigma*rdt*(y + K)/y
	S[1] = S[0]*np.exp(drift + vol*Z[1])
	for i in range(2,N+1):
		Z[i] = Z[i-1] - sigma*rdt*S[i-1]/(N*y)
		S[i] = S[i-1]*np.exp(drift + vol*Z[i-1])
	return (np.sum(S[1:])/N - K - y)
y = fsolve(func, K) # solve for y

# get the optimal drift vector
S = np.zeros(N+1) # store the price
mu = np.zeros(N+1)
mu[1] = sigma*rdt*(y + K)/y
S[0] = S0
S[1] = S[0]*np.exp(drift + vol*mu[1])
for i in range(2, N+1):
	mu[i] = mu[i-1] - sigma*rdt*S[i-1]/(N*y)
	S[i] = S[i-1] * np.exp(drift + vol * mu[i])
mu = mu[1:]

#print(mu) # we can get the same result as in GSH paper

pricepath = np.zeros((Num, N+1)) 
pricepath[:,0] = S0
dwpath = np.zeros((Num, N))
# generate normal under N(mu,I) joint normal distribution
#for i in range(N):
#	dwpath[:,i] = np.random.normal(mu[i],1,Num)
dwpath = np.random.normal(mu,1,(Num,N))

for i in range(1,N+1):
	pricepath[:,i] = pricepath[:,i-1]*(1 + r * dt + np.sqrt(V0*dt) * dwpath[:,i-1])

pricemean = np.sum(pricepath[:,1:], axis = 1)/N
payoff = np.array([ S - K if S - K > 0 else 0 for S in pricemean])
pricesample = [payoff[i]*np.exp(-np.dot(mu,dwpath[i,:]) + 0.5*np.dot(mu,mu)) for i in range(Num)]
price = np.mean(pricesample)*discount
stdev = np.std(pricesample)*discount
confi_L = price - 1.96 * stdev/np.sqrt(Num)
confi_H = price + 1.96 * stdev/np.sqrt(Num)
print("price of the asian option for constant volatility is :{:.2f} ".format(price))
print("95% confidence interval is : [ {:.5f},{:.5f} ]".format(confi_L,confi_H))
print("confidence interval spread is : {:.5f}".format(confi_H-confi_L))
print("standard deviation of the estimator is : {:.5f}".format(stdev/np.sqrt(Num)))
end = time.clock()
print("running time is: {:.2f} seconds".format(end - start))