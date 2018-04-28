# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:23:16 2017

@author: luodong
"""

import matplotlib.pyplot as plt

plt.plot(mu)
plt.axis([1,64,0,0.25])
plt.xlabel('drift vector index',fontsize = 16)
plt.ylabel('optimal value',fontsize = 16)
plt.title('Optimal drift vector for '+r'$\xi = 0$',fontsize = 16)
plt.tick_params(axis = "both", labelsize = 12)
plt.savefig("problemc_const_drift.png")