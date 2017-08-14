# -*- coding: utf-8 -*-



#### problem2 with control variable
#### we can create correlated brownian motions from independent brownian motions
import numpy as np
import time
from scipy.stats import norm 
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
W = A.dot(Z) # correlation brownian motion N*Num matrix

S = S0*np.exp(drift + vol*W) # N*Num matrix
#S_control = N*S0*np.exp(drift + vol*np.mean(W, axis = 0)) # control variable 1*Num matrix
S_control = N*np.power(np.prod(S, axis = 0), 1/N)

S_sum = np.sum(S, axis = 0) # 1*Num matrix
#### compute the analytical value for the control variable
#### control variable : N*S0*exp{(r-sigma^2/2)*T + sigma*(W1 + W2 + ... + WN)/N}
#### there is an average term in the control varialbe (W1 + W2 + ... + WN)/N
#### (W1 + W2 + ... + WN)/N is B*Z
#### is normal with mean 0, standard deviation sqrt(sum(B**2))
B = np.mean(A, axis = 0) # this is a row vector for summation
std_con = np.sqrt(B.dot(B)) # the standardeviation for the above average
sigma_c = sigma*std_con
#### the control variable can be regarded as a geometrical brownian motion
#### initial price N*S0, standard deviation sigma*std_con
#### compute the price from BS model
NewS0 = N*S0*np.exp(0.5*sigma**2*(std_con**2 - 1)*T)
#### BS model price is S(T) = NewS0*exp((r-0.5*sigma_c**2)*T + sigma_c*Z)
d1 = (np.log(NewS0/K) + (r + 0.5*sigma_c**2)*T)/(sigma_c*np.sqrt(T))
d2 = d1 - sigma_c*np.sqrt(T)
call_control = NewS0*norm.cdf(d1) - K*discount*norm.cdf(d2)
####  call option
callsample = [S - K if S - K > 0 else 0 for S in S_sum]
callsample_control = [S - K if S - K > 0 else 0 for S in S_control]
callprice = np.mean(np.array(callsample) - np.array(callsample_control)) * discount + call_control
callstdev = np.std(np.array(callsample) - np.array(callsample_control)) * discount
confi_L = callprice - 1.96 * callstdev/np.sqrt(Num)
confi_H = callprice + 1.96 * callstdev/np.sqrt(Num)
print("Result for simulation with control variable")
print("price of the option for is :{:.2f} ".format(callprice))
print("95% confidence interval is : [ {:.2f},{:.2f} ]".format(confi_L,confi_H))
print("confidence interval spread is : {:.5f}".format(confi_H-confi_L))
print("standard deviation for the estimator is :{:.5f}".format(callstdev/np.sqrt(Num)))
end = time.clock()
print("running time is: {:.2f} seconds".format(end - start))
