
import scipy.ndimage as nd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import filters, util, measure, morphology
from skimage import exposure
from matplotlib import path

# Function for saving image without frames or borders
def saveimage(fileName,fig=None,**kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
       Args:
            fileName (str): String that ends in .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain 
            aspect ratio.
    '''
    fig_size = fig.get_size_inches()
    w,h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
        w,h = kwargs['orig_size']
        w2,h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.xlim(0,h); plt.ylim(w,0)
    fig.savefig(fileName+'.png', dpi=300, transparent=True, bbox_inches='tight', \
                        pad_inches=0)	
	
# Hessian calculation function from John Fozard (JIC)
	
def hessian_ndimage(x, sigma=[1.0, 1.0]):
    N = x.ndim
    hessian = np.empty((N, N) + x.shape, dtype = x.dtype)
    for i in range(N):
        for j in range(N):
                u = [0]*N
                u[i] += 1
                u[j] += 1
                hessian[i,j,:,:] = nd.gaussian_filter(x, sigma, order=u)*sigma[i]*sigma[j]
    return hessian

# Eigenvalues calculation function from John Fozard (JIC)

def eigenvalues_2D_dir_abs(h):
    uxx = h[0,0,:,:]
    uxy = h[0,1,:,:]
    uyy = h[1,1,:,:]

    T = h[0,0,:,:] + h[1,1,:,:]
    D = h[0,0,:,:]*h[1,1,:,:] - h[1,0,:,:]*h[0,1,:,:]
    L1 = T/2.0 + np.sqrt(T*T/4.0-D)
    L2 = T/2.0 - np.sqrt(T*T/4.0-D)


    L1 = np.where(np.abs(L1)>np.abs(L2), L1, L2)
    L2 = np.where(np.abs(L1)>np.abs(L2), L2, L1)


    return L1, L2 #, e1, e2

# ACME line filter calculation function from John Fozard (JIC)

def acme_line_filter(im, s=1.0):
    h = hessian_ndimage(im.astype(float), sigma=[s, s])

    L1, L2 = eigenvalues_2D_dir_abs(h)

    S = np.sqrt(L1*L1+L2*L2)
    A = np.abs(L2)/(1e-8+np.abs(L1))
    gamma = 8.0
    alpha = 0.7
    c = 0.01

    P = np.where(L1<=0, (1.0 - np.exp(-S**2/2/gamma**2))*np.exp(-A**2/2/alpha**2)*np.exp(-2*c**2/(L1**2)), np.zeros_like(im))

    P = 255.0*P/np.max(P)

    return P

# ITK watershed segmentation from John Fozard (JIC)

def watershed(ma, # Input image intensity array
             r=2, # Radius for initial gaussian blur
             level=4, # Smaller level == more cells
             threshold=200, # Threshold image before processing
             minvolume=30, # Minimum cell area
             aniso=True): # Anisotropic diffusion (sharpen by try to keep edges

    ma_thresh = np.minimum(ma, threshold)
    itk_image = sitk.GetImageFromArray(ma_thresh)

    if aniso:
        itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)
        print "NL diffusion"
        itk_image = sitk.CurvatureAnisotropicDiffusion(itk_image)
        itk_image = sitk.Cast(itk_image, sitk.sitkInt16)
    if r>0:
        print "gaussian blur"
        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        gaussian_filter.SetVariance(r)
        itk_image = gaussian_filter.Execute(itk_image)

    ar1 = sitk.GetArrayFromImage(itk_image)

    print "watershed"
    itk_image = sitk.MorphologicalWatershed(itk_image, level=level, markWatershedLine=True, fullyConnected=True)
    print "relabel"
    itk_image = sitk.RelabelComponent(itk_image, minvolume)

    ar2 = sitk.GetArrayFromImage(itk_image)

    return ar1, ar2

def marker_watershed(ma, nuc,  # Input image intensity array
             r=2, # Radius for initial gaussian blur
             level=4, # Smaller level == more cells
             threshold=200, # Threshold image before processing
             minvolume=30, # Minimum cell area
             aniso=True): # Anisotropic diffusion (sharpen by try to keep edges

	ma_thresh = np.minimum(ma, threshold)
	itk_image = sitk.GetImageFromArray(ma_thresh)
	itk_nuc = sitk.GetImageFromArray(nuc)

	if aniso:
		itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)
		print "NL diffusion"
		itk_image = sitk.CurvatureAnisotropicDiffusion(itk_image)
		itk_image = sitk.Cast(itk_image, sitk.sitkInt16)
	if r>0:
		print "gaussian blur"
		gaussian_filter = sitk.DiscreteGaussianImageFilter()
		gaussian_filter.SetVariance(r)
		itk_image = gaussian_filter.Execute(itk_image)

		ar1 = sitk.GetArrayFromImage(itk_image)

	print "watershed"
	itk_image = sitk.MorphologicalWatershedFromMarkers(itk_image, itk_nuc, markWatershedLine=True, fullyConnected=False)
	
	print "relabel"
	#itk_image = sitk.RelabelComponent(itk_image, minvolume)

	ar2 = sitk.GetArrayFromImage(itk_image)

	return ar1, ar2

## SAVE image directly without imshow panel
def save(fileName, fig):
	fig_size = fig.get_size_inches()
	w,h = fig_size[0], fig_size[1]
	a=fig.gca()
	a.set_frame_on(False)
	a.set_xticks([]); a.set_yticks([])
	plt.axis('off')
	fig.savefig(fileName+'.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)