import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

r_with_error = .1 * np.random.rand(10) + 2
t = np.arange(0, 1.0, .1)
x = np.sin(2*np.pi*t)*r_with_error
x = np.append(x, [x[0]])
y = np.cos(2*np.pi*t)*r_with_error
y = np.append(y, [y[0]])
# x = np.sin(2*np.pi*t)
# y = np.cos(2*np.pi*t)
tck, u = interpolate.splprep([x, y], k=3, s=0)
unew = np.arange(0, 1.01, 0.01)
out = interpolate.splev(unew, tck)
plt.figure()
plt.plot(x, y, 'x', out[0], out[1], x, y, 'b')
plt.legend(['Linear', 'Cubic Spline'])
plt.axis([-2.1, 2.1, -2.1, 2.1])
plt.title('Spline of parametrically-defined curve')
plt.show()
