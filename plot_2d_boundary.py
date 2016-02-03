import matplotlib.pyplot as plt
from get_boundary_greyscale import get_boundary
import sys
import numpy as np

def plot(coords):
	x = []
	y = []
	for row in coords:
		x.extend([row[0][0], row[1][0]])
		y.extend([row[0][1], row[1][1]])
	plt.scatter(x, y, s=4, marker='.')
	plt.show()

th = int(sys.argv[1])
outer_boundary = get_boundary('q.png', th)
# print outer_boundary
plot(outer_boundary)
