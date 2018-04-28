# -*- coding: utf-8 -*-


### the numerical simulation here use monte carlo method with the risk neutral
### formula to price the asian option, Euler scheme is used for numerical integration
import numpy as np
r = 0.05 # risk neutral interest rate
S0 = 50   # initial underlying asset price
T = 1  # maturity of the option
N = 32
K = 50 # strike price
V0 = 0.3*0.3 # initial volatility factor
rho = 0.5 # correlation between two brownian motions
Num = 1000000 # number of paths to simulate
dt = T/N # 
ksi = 2
ksirdt = ksi * np.sqrt(dt)
ksi2dt = -0.5 * ksi**2*dt
### Correlated Brownian motions ###

### dw1 = dz1, dw2 = rho * dz1 + sqrt(1 - rho^2) * dz2 use this relation to 
### get two correlated Brownian motion, where Z1 and Z2 are two independent
### brownian motions respectively.

### Euler scheme for integration here ###

### S(t+1) = S(t) + r*S(t)*dt +sqrt(V(t))*S(t)*dW1(t)
### V(t+1) = V(t) + ksi*V(t)*dW2(t)

#(a)
### Note when ksi == 2, this corresponds to the stochasitic volatility case
### We only need to simulate two brownian motions together

pricepath = np.zeros((Num, N+1)) # price path for underlying asset
volpath = np.zeros((Num, N+1)) # stochastic volatility path 

# generate two independent random normal variables first
dz1 = np.random.normal(0, 1, (Num, N)) # generate the standard normal random variables
dz2 = np.random.normal(0, 1, (Num, N)) # generate the standard normal random variables


# create correlated brownian motions
dw1path = dz1
dw2path = rho * dz1 + np.sqrt(1 - rho**2) * dz2

pricepath[:, 0] = S0 # initial price for underlying asset
volpath[:, 0] = V0 # initial volatility

# do the Euler integration
for i in range(1,N+1):
	#pricepath[:,i] = pricepath[:,i-1] + r * pricepath[:,i-1] * dt + (np.sqrt(volpath[:,i-1]) * pricepath[:,i-1]*
	#dw1path[:,i-1] * np.sqrt(dt))
	#volpath[:,i] = volpath[:,i-1] + ksi * volpath[:,i-1] * dw2path[:,i-1] * np.sqrt(dt)
	pricepath[:,i] = pricepath[:,i-1] * ( 1 + r * dt + np.sqrt(volpath[:,i-1] * dt) * dw1path[:,i-1])
	volpath[:,i] = volpath[:,i-1] * np.exp(ksi2dt + ksirdt * dw2path[:,i-1])
	

pricemean = np.sum(pricepath[:,1:], axis = 1)/N 
payoff = [ S - K if S - K > 0 else 0 for S in pricemean]

price = np.mean(payoff) * np.exp(-r*T)

stdev = np.std(payoff) * np.exp(-r*T)
confi_L = price - 1.96 * stdev/np.sqrt(Num)
confi_H = price + 1.96 * stdev/np.sqrt(Num)
print("price of the asian option for stochastic volatility is :{:.2f} ".format(price))
print("95% confidence interval is : [ {:.2f},{:.2f} ]".format(confi_L,confi_H))
print("confidence interval spread is : {:.5f}".format(confi_H-confi_L))
print("standard deviation of the estimator is : {:.5f}".format(stdev/np.sqrt(Num)))