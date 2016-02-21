import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

r_with_error = .1 * np.random.rand(11) + 1
t = np.arange(0, 1.1, .1)
x = np.sin(2*np.pi*t)*r_with_error
y = np.cos(2*np.pi*t)*r_with_error
# x = np.sin(2*np.pi*t)
# y = np.cos(2*np.pi*t)
tck, u = interpolate.splprep([x, y], s=0)
unew = np.arange(0, 1.01, 0.01)
out = interpolate.splev(unew, tck)
plt.figure()
plt.plot(x, y, 'x', out[0], out[1], x, y, 'b')
plt.legend(['Linear', 'Cubic Spline'])
plt.axis([-1.05, 1.05, -1.05, 1.05])
plt.title('Spline of parametrically-defined curve')
plt.show()
