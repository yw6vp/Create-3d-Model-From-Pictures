from PIL import Image
import sys

def get_boundary(filename):
	im = Image.open(filename)
	im = im.filter(ImageFilter.FIND)