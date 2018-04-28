# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 21:43:42 2017

@author: luodong
"""

### importance sampling for stochastic volatility
import time
import numpy as np
import scipy.optimize as spo 
#from scipy.stats import norm
#import numdifftools as nd
import matplotlib.pyplot as plt
start = time.clock()
r = 0.05 # risk neutral interest rate
S0 = 50   # initial underlying asset price
T = 1  # maturity of the option
N = 32
K = 50 # strike price
V0 = 0.3*0.3 # initial volatility factor
rho = 0.5 # correlation between two brownian motions
Num = 100000 # number of paths to simulate
dt = T/N #
rdt = np.sqrt(dt)
sigma = np.sqrt(V0)
drift = (r - 0.5*sigma**2)*dt
vol = sigma*rdt
discount = np.exp(-r*T)
ksi = 2
ksirdt = ksi * np.sqrt(dt)
ksi2dt = -0.5 * ksi**2*dt

## we need to use numerical optimization to solve for mu when we have stochastic volatility
## max F(z) - 1/2 * Z.dot(Z)
def F(z):
	S = np.zeros(N+1)
	V = np.zeros(N+1)
	S[0] = S0
	V[0] = V0
	for i in range(1, N+1):
		S[i] = S[i-1] * ( 1 + r * dt + np.sqrt(V[i-1] * dt) * z[i-1])
		V[i] = V[i-1] * np.exp(ksi2dt + ksirdt * z[N+i-1])
	return np.log(discount*(np.mean(S[1:])-K))
## minimize this function to get the optimal drift vector	
def func(z):
	return -F(z)+0.5*np.dot(z,z)
	
#opt1 = spo.brute(-func, tuple(zip([-0.5]*(2*N),[0.5]*(2*N),[0.25]*(2*N))),finish = None)
# optimal drift vector
opt1 = spo.fmin(func, [0.3]*64, xtol = 0.0005, ftol = 0.0005, maxiter = 100000)#, maxfun = 20)
#==============================================================================
# plt.plot(opt1,linewidth = 2,label = 'optimal drift vector')
# #plt.plot(-opteigenvec,linewidth = 2,label = 'optimal eigenvector') # note that I need to change the sign of opteigenvec to compare.
# plt.legend()
# plt.axis([1,64,-0.1,0.35])
# plt.xlabel('drift vector index',fontsize = 16)
# plt.ylabel('optimal value',fontsize = 16)
# plt.title('Optimal drift vector for '+r'$\xi = 2$',fontsize = 16)
# plt.tick_params(axis = "both", labelsize = 12)
# plt.savefig("problemc_stoch_drift.png")
#==============================================================================
mu = opt1 # optimal drift vector

pricepath = np.zeros((Num, N+1)) 
pricepath[:,0] = S0
volpath = np.zeros((Num, N+1)) # stochastic volatility path
volpath[:, 0] = V0 # initial volatility
dwpath = np.zeros((Num, N))
#generate normal under N(mu,I) joint normal distribution
dwpath = np.random.normal(mu,1,(Num,2*N))

for i in range(1,N+1):
	pricepath[:,i] = pricepath[:,i-1] * ( 1 + r * dt + np.sqrt(volpath[:,i-1] * dt) * dwpath[:,i-1])
	volpath[:,i] = volpath[:,i-1] * np.exp(ksi2dt + ksirdt * dwpath[:,N+i-1])
	
pricemean = np.sum(pricepath[:,1:], axis = 1)/N
payoff = np.array([ S - K if S - K > 0 else 0 for S in pricemean])
pricesample = [payoff[i]*np.exp(-np.dot(mu,dwpath[i,:]) + 0.5*np.dot(mu,mu)) for i in range(Num)]
price = np.mean(pricesample)*discount
stdev = np.std(pricesample)*discount
confi_L = price - 1.96 * stdev/np.sqrt(Num)
confi_H = price + 1.96 * stdev/np.sqrt(Num)


#########################
print("price of the asian option for stochastic volatility is :{:.2f} ".format(price))
print("95% confidence interval is : [ {:.5f},{:.5f} ]".format(confi_L,confi_H))
print("confidence interval spread is : {:.5f}".format(confi_H-confi_L))
print("standard deviation of the estimator is : {:.5f}".format(stdev/np.sqrt(Num)))
end = time.clock()
print("running time is: {:.2f} seconds".format(end - start))