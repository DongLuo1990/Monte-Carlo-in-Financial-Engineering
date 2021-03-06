# -*- coding: utf-8 -*-

### particularly, this does the antithetic variable test

### there are different ways to improve the results in (a)

### 1. increase the number of simulation path Num
### 2. decrease the time step size dt for the numerical integration of the path
### 3. antithetic sampling to decrease the variance, a variance reduction technique

### use the antithetic sampling ######


### the numerical simulation here use monte carlo method with the risk neutral
### formula to price the asian option, Euler scheme is used for numerical integration
import time
import numpy as np

start = time.clock()
r = 0.05 # risk neutral interest rate
S0 = 50   # initial underlying asset price
T = 1  # maturity of the option
#n = 128 # number of time steps where to compute the average
#stepfactor = 1 # note that the number of numerical integration steps can be different from summation points
#N = 128*stepfactor # number of time steps for numerical integration, can be different from n 
N = 32 
K = 50 # strike price
V0 = 0.3*0.3 # initial volatility factor
rho = 0.5 # correlation between two brownian motions
Num = 1000000 # number of paths to simulate
dt = T/N # 
### Correlated Brownian motions ###

### dw1 = dz1, dw2 = rho * dz1 + sqrt(1 - rho^2) * dz2 use this relation to 
### get two correlated Brownian motion, where Z1 and Z2 are two independent
### brownian motions respectively.

### Euler scheme for integration here ###

### S(t+1) = S(t) + r*S(t)*dt +sqrt(V(t))*S(t)*dW1(t)
### V(t+1) = V(t) + ksi*V(t)*dW2(t)

#(a)
### Note when ksi == 0, V(t) == V(0), this corresponds to the constant volatility case
### We only need to simulate one brownian motion

pricepath1 = np.zeros((Num, N+1)) # price path for underlying asset
pricepath2 = np.zeros((Num, N+1)) # dual price path for underlying asset
dwpath = np.zeros((Num, N))  # Brownian motion

dwpath = np.random.normal(0, 1, (Num, N)) # generate the standard normal random variables

pricepath1[:,0] = S0 # initial price for underlying asset
pricepath2[:,0] = S0 # dual path of 1

### Maybe what I do here is just not the right way.
for i in range(1,N+1):
	#pricepath[:,i] = pricepath[:,i-1] + r * pricepath[:,i-1] * dt + (np.sqrt(V0) * pricepath[:,i-1]*
	#dwpath[:,i-1]*np.sqrt(dt))
	pricepath1[:,i] = pricepath1[:,i-1]*( 1 + r * dt + np.sqrt(V0*dt) * dwpath[:,i-1])
	pricepath2[:,i] = pricepath2[:,i-1]*( 1 + r * dt - np.sqrt(V0*dt) * dwpath[:,i-1])
	
#sumindex = np.arange(stepfactor, N+1, stepfactor) # sum at the required time points
#pricepath = (pricepath1 + pricepath2) / 2
#pricepath = np.vstack((pricepath1, pricepath2))
#pricemean = np.sum(pricepath[:,1:], axis = 1)/N 
pricemean1 = np.sum(pricepath1[:,1:], axis = 1)/N 
pricemean2 = np.sum(pricepath2[:,1:], axis = 1)/N
payoff1 = [ S - K if S - K > 0 else 0 for S in pricemean1]
payoff2 = [ S - K if S - K > 0 else 0 for S in pricemean2]
payoff = (np.array(payoff1)+np.array(payoff2))/2

price = np.mean(payoff) * np.exp(-r*T)
#price = np.mean(pricemean - K)*np.exp(-r*T)
#price = (price if price > 0 else 0)
stdev = np.std(payoff) * np.exp(-r*T)
confi_L = price - 1.96 * stdev/np.sqrt(Num)
confi_H = price + 1.96 * stdev/np.sqrt(Num)
print("price of the asian option for constant volatility is :{:.2f} ".format(price))
print("95% confidence interval is : [ {:.2f},{:.2f} ]".format(confi_L,confi_H))
print("confidence interval spread is : {:.3f}".format(confi_H-confi_L))
print("standard deviation of the estimator is : {:.5f}".format(stdev/np.sqrt(Num)))
end = time.clock()
print("running time is: {:.2f} seconds".format(end - start))