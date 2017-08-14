# -*- coding: utf-8 -*-

### problem1
### use antithetic variables here !
### the random process that drives the underlying S(t) is a linear stochastic
### equation, it's easy to integrate to get a formula for S(t), and we can see
### S(T) is a normal random variable
import time
import numpy as np
start = time.clock()
print("Simulation with antithetic sampling")
#### parameters
a = 0.5
b = 104
r = 0.05
sigma = 10
T = 1
S0 = 100
K = 105
Num = 1000000 # number of simulation samples
discount = np.exp(-r*T)
#### compute the mean and standard deviation for S(T)

mu = np.exp(-a*T)*S0 + b*(1 - np.exp(-a*T))
Sigma = np.sqrt(sigma**2*(1 - np.exp(-2*a*T))/(2*a))

#### generate the random sample
randomsample = np.random.normal(0, Sigma, Num)
pricesample = randomsample + mu
pricesample_anti = - randomsample + mu # the antithetic sample

#### put option
putsample = [K - S if K - S > 0 else 0 for S in pricesample]
putsample_anti = [K - S if K - S > 0 else 0 for S in pricesample_anti]
putsample = (np.array(putsample) + np.array(putsample_anti)) / 2
putprice = np.mean(putsample) * discount
putstdev = np.std(putsample) * discount
confi_L = putprice - 1.96 * putstdev/np.sqrt(Num)
confi_H = putprice + 1.96 * putstdev/np.sqrt(Num)
print("Result for question (a)")
print("price of the put option for is :{:.2f} ".format(putprice))
print("95% confidence interval is : [ {:.2f},{:.2f} ]".format(confi_L,confi_H))
print("confidence interval spread is : {:.5f}".format(confi_H-confi_L))
print("standard deviation for the estimator is :{:.5f}".format(putstdev/np.sqrt(Num)))



#### option in (b)
putsample = [K - S if K - S > 0 else S - K for S in pricesample]
putsample_anti = [K - S if K - S > 0 else S - K for S in pricesample_anti]
putsample = (np.array(putsample) + np.array(putsample_anti)) / 2
putprice = np.mean(putsample) * discount
putstdev = np.std(putsample) * discount
confi_L = putprice - 1.96 * putstdev/np.sqrt(Num)
confi_H = putprice + 1.96 * putstdev/np.sqrt(Num)
print("Result for question (b)")
print("price of the option with payoff in (b) for is :{:.2f} ".format(putprice))
print("95% confidence interval is : [ {:.2f},{:.2f} ]".format(confi_L,confi_H))
print("confidence interval spread is : {:.5f}".format(confi_H-confi_L))
print("standard deviation for the estimator is :{:.5f}".format(putstdev/np.sqrt(Num)))
end = time.clock()
print("running time is: {:.2f} seconds".format(end - start))