from get_boundary_greyscale import get_boundary
import sys
import numpy as np

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
	N = 20
	# Initialize x, y and z with NAN and height rows and N cols
	x = np.empty((height, N))
	x[:] = np.NAN
	y = np.empty((height, N))
	y[:] = np.NAN
	z = np.empty((height, N))
	z[:] = np.NAN
	for i in range(height):
		for j in range(N):
			theta = np.pi * 2 * j / N
			x[i][j] = x_original[i] * np.cos(theta)
			y[i][j] = y_original[i] * np.cos(theta)
			z[i][j] = i
	return x, y, z













