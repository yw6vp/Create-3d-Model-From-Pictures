from PIL import Image
from PIL import ImageFilter
import sys

def get_boundary(filename, th):
	# Open image
	im = Image.open(filename)
	width, height = im.size
	# Convert to greyscale
	mask = im.convert('L')
	# Convert to black and white
	mask = mask.point(lambda i : i < th and 255)
	# mask.show()
	# Find edges
	mask = mask.filter(ImageFilter.FIND_EDGES)
	# cropping off the edges of the image because their values are also 255
	mask = mask.crop((10, 2, width - 10, height - 2))
	# mask.show()
	# return only the outer boundary pixels
	width, height = mask.size
	boundary = list(mask.getdata())
	outer_boundary = []
	for row in xrange(height):
		outer_boundary.append([])
		for col in xrange(width):
			# [x, y]
			coord = [col, row] 
			if boundary[row * width + col] == 255:
				if len(outer_boundary[row]) == 0:
					outer_boundary[row] = [coord]
				elif len(outer_boundary[row]) == 1:
					outer_boundary[row].append(coord)
				else:
					outer_boundary[row][1] = coord

	return outer_boundary

# th = int(sys.argv[1])
# print th, type(th)
# outer_boundary = get_boundary('q.png', th)
# print outer_boundary