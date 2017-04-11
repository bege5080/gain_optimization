import itertools
import numpy as np
from scipy.ndimage.interpolation import rotate, shift
from scipy.special import gamma,hyp2f1


def polar_grid(imagepix,pupilpix):
	'''
	make a polar image grid from a cartesian grid
	'''
	grid=np.mgrid[0:imagepix,0:imagepix]
	xy=np.sqrt((grid[0]-imagepix/2.)**2.+(grid[1]-imagepix/2.)**2.)
	xy[np.where(xy>pupilpix/2.)]=0.
	rad_norm=xy/np.max(xy)
	phi=np.arctan2(grid[1]-imagepix/2.,grid[0]-imagepix/2.)
	return rad_norm,phi

def zernike(n,m,rho,phi):
	'''
	make a zernike polynomial of specified n,m given input polar coordinate maps of rho (normalized to one; pupil coordinates only) and phi (radians)
	'''

	rad=gamma(n+1)*hyp2f1(-1./2.*(m+n),1./2.*(m-n),-n,rho**(-2))/gamma(1./2.*(2+n-m))/gamma(1./2.*(2+n+m))*rho**n
	if m>=0:
		cos=np.cos(m*phi)
		out=rad*cos
	else:
		sin=np.sin(-1*m*phi)
		out=rad*sin
	out[np.where(np.isnan(out)==True)]=0.
	return out

def zern_covinv(imagepix,pupilpix):

	rho,phi=polar_grid(imagepix,pupilpix)

	zern_nm=[]
	for n in range(1,6): #remove zernikes up to n=5, which is the first 21 modes
		m=range(-1*n,n+2,2)
		for mm in m:
			zern_nm.append([n,mm])

	#reference array
	refarr=np.zeros((len(zern_nm),imagepix**2))
	for i in range(len(zern_nm)):
		z=zernike(zern_nm[i][0],zern_nm[i][1],rho,phi)
		refarr[i]=z.flatten()

	#covariance matrix:
	n=len(zern_nm)
	cov=np.zeros((n,n))
	for i in range(n):
		for j in range(i+1):
			if cov[j,i]==0.:
				cov[i,j]=np.sum(refarr[i,:]*refarr[j,:])
				cov[j,i]=cov[i,j]
			#print i*n+j,n**2-1
	covinv=np.linalg.pinv(cov,rcond=1e-7)

	return covinv,n,refarr

def zern_lsq(imtar,imagepix,pupilpix,covinv,n,refarr):

	#correlation image vector:
	tar=np.ndarray.flatten(imtar)
	cor=np.zeros((n,1))
	for i in range(n):
		cor[i]=np.sum(refarr[i]*tar)
		#print i, n-1

	coeffs=np.dot(covinv,cor)

	return coeffs