from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

x=np.array([0,1])
y=np.array([3.5, 7])
cs = CubicSpline(x,y, bc_type='clamped')
xs = np.array([a/10 for a in range(11)])
ys = cs(xs)
print (xs)
print (ys)
plt.plot(xs, ys,'o')
plt.show()