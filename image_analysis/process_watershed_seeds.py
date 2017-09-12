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
	refchannel = int(sys.argv[2])-1 
	meschannel = int(sys.argv[3])-1 
	memchannel = int(sys.argv[4])-1 
	sel = 3
	gau = 6
	acm = 3
	sig = int(sys.argv[5])
	
	## Load TIF file
	data = tif.imread(sys.argv[1])
	ref = data[refchannel].astype(np.uint16)
	mes = data[meschannel].astype(np.uint16)
	mem = data[memchannel].astype(float)
	
	# Generating summed nuclei array
	total = np.minimum(ref + mes, 255)
	total = total.astype(np.uint8)
	
	## Rescale intensity of pixel values.
	
	# Use a percentile for rescaling
	v_min, v_max = np.percentile(ref, (2, 98))
	#rescaled = exposure.rescale_intensity(ref, in_range=(v_min, v_max))
	rescaled = exposure.rescale_intensity(total, in_range=(v_min, v_max))

	## Perform median filtering to remove noise
	ref_noise = filters.rank.median(rescaled, morphology.disk(2))
	
	## Otsu threshold calculation
	thresh = filters.threshold_otsu(ref_noise)
	
	## Binarisation using calculated threshold
	binary = ref_noise > thresh
	
	## Label connected features
	nuclei = measure.label(binary)
	nuc = nuclei.astype(np.uint16)
	print np.max(nuc)
	
	## Membrance processing
	
	# From John
	median = nd.median_filter(mem, sel) # Median filter to reduce shot noise
	local = np.maximum(median - nd.gaussian_filter(median, gau), 0)# try to enhance local contrast
	im2 = acme_line_filter(local, s=acm)	
	# Call segmentation routine
	
	im3, seg = marker_watershed(im2, nuc, r=2, level=sig, aniso=True)
		
	#cmap = 0.75*npr.random((np.max(seg)+1,3))+0.25
	#seg2 = cmap[seg, :]
	
	# To generate better differentiation with black (0 values)
	
	#rel = rel + 20
	
	#rel[rel == 21] = 0
		

	img = np.zeros_like(mem)
	
	## RELABELLING BY AREA
	#for region in measure.regionprops(seg):
	#	if (region.area > 2000 or region.area < 30):
	#		print "out of bounds",region.label,region.area
	#	else:
	#		if (np.mean(ref[seg == region.label]) < np.mean(mes[seg == region.label])):
	#			print region.area, np.mean(ref[seg == region.label]), np.mean(mes[seg == region.label])
	#			for coord in region.coords:
	#				img[coord[0],coord[1]] = region.area
			
	## RELABELLING BY NUCLEAR VALUE
	seg = cleanup(seg)
	for i in range(1,np.max(seg)):
		img[seg==i] = np.mean(mes[nuc==i])

	## Save figure
	plt.ioff()
	fig = plt.figure(figsize=(4.5,4.5))
	plt.imshow(img, cmap="inferno")
	plt.colorbar()
	plt.imshow(binary, cmap=plt.cm.gray, alpha=0.3)
	plt.show()
	saveimage("2-mes_inferno"+str(sig), fig)
	plt.close()

if len(sys.argv) < 5:
	sys.exit("###\n\nUsage: python process_marker_watershed.py image [RefCh] [MesCh] [MemCh] [Param]\n\n###")
main()
