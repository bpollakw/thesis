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
	
	## Load TIF file
	data = tif.imread(sys.argv[1])
	ref = data[refchannel].astype(np.uint16)
	mes = data[meschannel].astype(np.uint16)
	
	
	total = np.minimum(ref + mes, 255)
	#total = np.maximum(ref - mes, 0)
	total = total.astype(np.uint8)
	
	
	## Rescale intensity of pixel values.
	
	# Use a percentile for rescaling
	v_min, v_max = np.percentile(ref, (2, 98))
	#rescaled = exposure.rescale_intensity(ref, in_range=(v_min, v_max))
	rescaled = exposure.rescale_intensity(total, in_range=(v_min, v_max))
	
	## Add noise to perform median filtering
	noisy = rescaled
	noise = np.random.random(rescaled.shape)
	noisy[noise > 0.98] = 255
	noisy[noise < 0.02] = 0

	## Perform median filtering to remove noise
	ref_noise = filters.rank.median(noisy, morphology.disk(2))
	ref_noise = morphology.closing(ref_noise)
	#ref_noise = filters.gaussian(ref_noise, sigma=1)
	
	## Otsu threshold calculation
	thresh = filters.threshold_otsu(ref_noise)
	
	## Binarisation using calculated threshold
	binary = ref_noise > thresh
	
	## Label connected features
	nuclei = measure.label(binary)
	nuc = nuclei.astype(np.uint16)
	print np.max(nuc)
	rat = nuclei
	#for i in range(1,np.max(nuc)+1):
		#if np.mean(mes[nuc == i])-np.mean(ref[nuc == i]) < 0:
			#print np.mean(ref[nuc == i]),np.mean(mes[nuc == i])
		#	rat[nuc == i] = 0
		#else:
		#	print np.mean(ref[nuc == i]),np.mean(mes[nuc == i])
		#	rat[nuc == i] = 255
	#	rat[nuc == i] = np.mean(mes[nuc == i])/np.mean(ref[nuc == i])
	#quantified = (mes )/(ref + 1)
	#quantified = (mes/np.minimum(ref+1,255)) * binary

	## Save figure
	#plt.ioff()
	fig = plt.figure(figsize=(4.5,4.5))
	plt.imshow(rat, cmap="nipy_spectral")
	#plt.imshow(total, cmap=plt.cm.gray)
	plt.colorbar()
	#plt.clim(0,2.5)
	plt.show()
	saveimage("4_nuc_total", fig)
	plt.close()

if len(sys.argv) < 3:
	sys.exit("###\n\nUsage: python process_nuclei.py image [RefChannel] [MesChannel]\n\n###")
main()
