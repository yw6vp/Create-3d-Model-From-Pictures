from get_boundary_greyscale import get_boundary
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
def make_3d_model(boundary):
	height = len(boundary)
	x_original, y_original = transform(boundary)
	# Total number of cross sections
	N = 10
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

def plot_3d_model(x, y, z):
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

th = int(sys.argv[1])
boundary = get_boundary('q.png', th)
x, y, z = make_3d_model(boundary)
plot_3d_model(x, y, z)














