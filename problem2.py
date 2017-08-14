# -*- coding: utf-8 -*-

#### problem2 without control variable
#### we can create correlated brownian motions from independent brownian motions
import numpy as np
import time
start = time.clock()
#### parameters
m = 10
r = 0.04
sigma = 0.15
T = 1
rho = 0.2
N = 10 # Number of underlying asset
S0 = 100
K = S0*N
Num = 1000000 # Number of simulation path
#### precompute
drift = (r - 0.5*sigma**2)*T
vol   = sigma*np.sqrt(T)
discount = np.exp(-r*T)

C = np.ones((N,N))*rho # correlation matrix
for i in range(N):
	C[i,i] = 1

A = np.linalg.cholesky(C) # cholesky decomposition to get AA' = C

### rho_i,k = sum(a_i,j*a_k,j) from j = 1 to j = N
### Wi = sum(a_i,j*Zj) from j = 1 to j = N, Zj is independent Brownian motion,
### Wi is the correlated brownian motion

### rho is constant here, therefore A is constant matrix
### we have the relation as W = A*Z, W is a column vector, Z is also a column 
### vector
Z = np.random.normal(0, 1, (N, Num)) # independent normal
W = A.dot(Z) # correlation brownian motion

S = S0*np.exp(drift + vol*W)

S_sum = np.sum(S, axis = 0)

####  call option
callsample = [S - K if S - K > 0 else 0 for S in S_sum]
callprice = np.mean(callsample) * discount
callstdev = np.std(callsample) * discount
confi_L = callprice - 1.96 * callstdev/np.sqrt(Num)
confi_H = callprice + 1.96 * callstdev/np.sqrt(Num)
print("Result for simulation without control variable")
print("price of the option for is :{:.2f} ".format(callprice))
print("95% confidence interval is : [ {:.2f},{:.2f} ]".format(confi_L,confi_H))
print("confidence interval spread is : {:.5f}".format(confi_H-confi_L))
print("standard deviation for the estimator is :{:.5f}".format(callstdev/np.sqrt(Num)))
end = time.clock()
print("running time is: {:.2f} seconds".format(end - start))
