from get_boundary_greyscale import get_boundary
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random
from scipy import interpolate

def transform(boundary):
	x = []
	y = []
	for row in boundary:
		x.extend([row[0][0], row[1][0]])
		y.extend([row[0][1], row[1][1]])
	x = np.array(x)
	y = np.array(y)
	x_center = np.mean(x)
	y_center = np.mean(y)
	x -= x_center
	y -= y_center
	return x, y

# Temporary version, make 3d model by rotating the same cross section
def make_3d_model_naive(boundary):
	height = len(boundary)
	x_original, y_original = transform(boundary)
	# Total number of cross sections
	N = 2
	# Initialize x, y and z with NAN and height rows and 2 * N cols
	x = np.empty((height, 2 * N))
	x[:] = np.NAN
	y = np.empty((height, 2 * N))
	y[:] = np.NAN
	z = np.empty((height, 2 * N))
	z[:] = np.NAN
	for i in range(height):
		for j in range(N):
			theta = np.pi * j / N
			x[i][2 * j] = x_original[2 * i] * np.cos(theta)
			x[i][2 * j + 1] = x_original[2 * i + 1] * np.cos(theta)
			y[i][2 * j] = x_original[2 * i] * np.sin(theta)
			y[i][2 * j + 1] = x_original[2 * i + 1] * np.sin(theta)
			z[i][2 * j] = y_original[i * 2]
			z[i][2 * j + 1] = y_original[i * 2 + 1]
	return x, y, z

def make_3d_model(boundary):
	height = len(boundary)
	x_original, y_original = transform(boundary)
	# Total number of cross sections
	N = 10
	# Initialize x, y and z with NAN and height rows and 2 * N cols
	x = np.empty((height, 2 * N + 1))
	x[:] = np.NAN
	y = np.empty((height, 2 * N + 1))
	y[:] = np.NAN
	z = np.empty((height, 2 * N + 1))
	z[:] = np.NAN
	for i in range(height):
		for j in range(N):
			theta = np.pi * j / N
			factor = 1 + .02 * random()
			x[i][j] = x_original[2 * i] * np.cos(theta) * factor
			x[i][N + j] = x_original[2 * i + 1] * np.cos(theta) * factor
			y[i][j] = x_original[2 * i] * np.sin(theta) * factor
			y[i][N + j] = x_original[2 * i + 1] * np.sin(theta) * factor
			z[i][j] = y_original[i * 2]
			z[i][N + j] = y_original[i * 2 + 1]
		x[i][2 * N] = x[i][0]
		y[i][2 * N] = y[i][0]
		z[i][2 * N] = z[i][0]
	# smooth_one_layer(x[100], y[100])

	layers = []
	for i in range(height):
		tck, u = interpolate.splprep([x[i], y[i]], s=0)
		unew = np.arange(0, 1.01, 0.01)
		out = interpolate.splev(unew, tck)
		layers.append([out[0], out[1]])
	return x, y, z, layers
	# return x, y, z

def smooth_one_layer(x, y):
	tck, u = interpolate.splprep([x, y], s=0)
	unew = np.arange(0, 1.01, 0.01)
	out = interpolate.splev(unew, tck)
	plt.figure()
	plt.plot(x, y, 'x', out[0], out[1], x, y, 'b')
	plt.legend(['Linear', 'Cubic Spline'])
	# plt.axis([-2.1, 2.1, -2.1, 2.1])
	plt.title('Spline of parametrically-defined curve')
	plt.show()

def plot_3d_model_naive(x, y, z):
	# Convert x, y, z to 1d array
	flat_x = x.flatten()
	flat_y = y.flatten()
	flat_z = z.flatten()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(flat_x, flat_y, flat_z, s=4, marker='.')
	# plt.axis([-500, 500, -500, 500])
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()

def plot_one_layer(x, y, model, index):
	layer = model[index]
	plt.figure()
	plt.plot(x[index], y[index], 'x', layer[0], layer[1], x[index], y[index], 'b')
	plt.legend(['Linear', 'Cubic Spline'])
	# plt.axis([-2.1, 2.1, -2.1, 2.1])
	plt.title('Spline of parametrically-defined curve')
	plt.show()

th = int(sys.argv[1])
boundary = get_boundary('q.png', th)
# x, y, z = make_3d_model_naive(boundary)
# plot_3d_model_naive(x, y, z)
# x, y, z = make_3d_model(boundary)
# plot_3d_model_naive(x, y, z)
x, y, z, model = make_3d_model(boundary)
plot_one_layer(x, y, model, 100)















