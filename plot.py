# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

arr = np.asarray
### problem1 a1
#==============================================================================
# x = arr([1,2,3,4])
# y = arr([5.06,4.83,4.95,4.92])
# c = arr([[4.72,5.39],[4.72,4.93],[4.92,4.98],[4.91,4.93]])
#==============================================================================
### problem1 a2
#==============================================================================
# x = arr([1,2,3,4])
# y = arr([4.93,4.94,4.92,4.92])
# c = arr([[4.81,5.06],[4.90,4.98],[4.90,4.93],[4.92,4.93]])
#==============================================================================
## problem b1
#==============================================================================
# x = arr([1,2,3,4])
# y = arr([6.68,6.55,6.60,6.59])
# c = arr([[6.38,6.99],[6.45,6.64],[6.57,6.63],[6.58,6.60]])
#==============================================================================
#==============================================================================
# x = arr([1,2,3,4])
# y = arr([6.61,6.62,6.57,6.58])
# c = arr([[6.35,6.86],[6.54,6.70],[6.55,6.60],[6.58,6.59]])
#==============================================================================
#==============================================================================
# # problem2a
# x = arr([1,2,3,4])
# y = arr([56.47,54.45,54.71,54.51])
# c = arr([[52.54,60.39],[53.24,55.66],[54.34,55.09],[54.39,54.62]])
# yerr = (c[:,1]-c[:,0])/2
# # problem2a
# x1 = arr([1,2,3,4])
# y1 = arr([54.52,54.49,54.54,54.52])
# c1 = arr([[54.20,54.85],[54.39,54.59],[54.51,54.57],[54.51,54.53]])
# yerr1 = (c1[:,1]-c1[:,0])/2
#==============================================================================
#problemb1
x = arr([1,2,3,4])
y = arr([3.66,3.74,3.81,3.84])
c = arr([[3.18,4.14],[3.56,3.91],[3.74,3.88],[3.81,3.88]])
yerr = (c[:,1]-c[:,0])/2
#problemb2
#==============================================================================
# x = arr([1,2,3,4])
# y = arr([3.66,3.74,3.81,3.84])
# c = arr([[3.18,4.14],[3.56,3.91],[3.74,3.88],[3.81,3.88]])
#==============================================================================
#==============================================================================
# x1 = arr([1,2,3,4])
# y1 = arr([4.17,4.05,4.07,4.08])
# c1 = arr([[3.97,4.37],[3.99,4.12],[4.05,4.09],[4.07,4.08]])
# yerr1 = (c1[:,1]-c1[:,0])/2
#==============================================================================
x1 = arr([1,2,3,4])
y1 = arr([3.80,3.86,3.86,3.87])
c1 = arr([[3.65,3.95],[3.81,3.91],[3.84,3.88],[3.87,3.88]])
yerr1 = (c1[:,1]-c1[:,0])/2
									
x2 = arr([1,2,3,4])
y2 = arr([3.82,3.89,3.87,3.87])
c2 = arr([[3.704,3.934],[3.844,3.937],[3.860,3.887],[3.865,3.873]])
yerr2 = (c2[:,1]-c2[:,0])/2									
									
plt.scatter(x,y,color = "red",s=60)
plt.errorbar(x, y, yerr=yerr, zorder=0, fmt="none",
             marker="none",ecolor = "red", linewith = 10,capsize = 8,capthick = 2)
plt.scatter(x1,y1,color = "blue",s=60)
plt.errorbar(x1, y1, yerr=yerr1, zorder=0, fmt="none",
             marker="none",ecolor = "blue", linewith = 10,capsize = 8,capthick = 2)
plt.scatter(x2,y2,color = "green",s=60)
plt.errorbar(x2, y2, yerr=yerr2, zorder=0, fmt="none",
             marker="none",ecolor = "green", linewith = 10,capsize = 8,capthick = 2)
#plt.axhline(y=4.07, xmin=0, xmax=1)
plt.axis([0.5,4.5,3.1,4.2])
plt.xlabel('Number of simulation',fontsize = 16)
plt.ylabel('option price',fontsize = 16)
plt.title('Simulation for '+r'$\xi = 2$',fontsize = 16)
plt.xticks([1,2,3,4])
ax = plt.gca()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = '1000'
labels[1] = '10000'
labels[2] = '100000'
labels[3] = '1000000'
ax.set_xticklabels(labels)
plt.tick_params(axis = "both", labelsize = 16)
plt.legend(['Without Variance Reduction','Importance Sampling','IS & Strafify sampling'],loc = 4)
plt.savefig("problemc_stocha_comp.png")
