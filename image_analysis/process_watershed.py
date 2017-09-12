import sys
import tifffile as tif
import scipy.ndimage as nd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from skimage import filters, util, measure, morphology
from skimage import exposure
from matplotlib import path
from os import path
from helper import *

def main():
	## Parameters 
	memchannel = int(sys.argv[2])-1 
	sel = 3
	gau = 6
	acm = 3
	sig = int(sys.argv[3])
	
	## Load TIF file
	data = tif.imread(sys.argv[1])
	
	im = data[memchannel].astype(float)
	
	# From John
	median = nd.median_filter(im, sel) # Median filter to reduce shot noise
	local = np.maximum(median - nd.gaussian_filter(median, gau), 0)# try to enhance local contrast
	im2 = acme_line_filter(local, s=acm)	
	# Call segmentation routine
	im3, seg = watershed(im2, r=2, level=sig, aniso=True)
		
	#cmap = 0.75*npr.random((np.max(seg)+1,3))+0.25
	#seg2 = cmap[seg, :]
	
	# To generate better differentiation with black (0 values)
	rel = seg
	rel = rel + 20
	
	rel[rel == 21] = 0
		
	print " "
	print "Segmented number:"+str(np.max(seg))
	## Save figure
	plt.ioff()
	fig = plt.figure(figsize=(4.5,4.5))
	plt.imshow(rel, cmap="nipy_spectral")
	save("seg_s"+str(sig), fig)
	plt.close()

if len(sys.argv) < 3:
	sys.exit("###\n\nUsage: python process_watershed_membrane.py [IMAGE] [OUTLINE_CHANNEL] [LEVEL]\n\n###")
main()
