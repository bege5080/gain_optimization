'''
all the transfer functions and how to get their magnitude and phase
'''

from matplotlib.widgets import Slider, Button, RadioButtons
import os
import sys
import itertools
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate
import scipy.ndimage.interpolation as inter
from astropy.convolution import convolve,convolve_fft
import scipy

#magnitude and phase of transfer functions:
mag = lambda tf: 20.*np.log10(np.sqrt(np.real(tf)**2.+np.imag(tf)**2.)) #magnitude in dB
phase = lambda tf: np.unwrap(np.arctan2(np.imag(tf),np.real(tf)))*180./np.pi #np.unwrap prevents 2 pi phase discontinuities
s2f = lambda f: 1.j*2.*np.pi*f

#transfer functions
'''
H_wfs = lambda s,T_s: (1.-np.exp(-1.*T_s*s))/(T_s*s)

H_lag = lambda s,tau: np.exp(-1.*tau*s)

H_int = lambda s,T_s: 1./(1.-np.exp(-1.*T_s*s))

H_cont = lambda g,s,T_s: g*H_int(s,T_s)

H_ol_splane = lambda g,s,T_s,tau: H_wfs(s,T_s)*H_lag(s,tau)*H_cont(g,s,T_s)
H_ol = lambda g,f,T_s,tau: H_ol_splane(g,s2f(f),T_s,tau)

H_rej = lambda g,f,T_s,tau: 1./(1.+ H_ol(g,f,T_s,tau))

H_cl = lambda g,f,T_s,tau: H_ol(g,f,T_s,tau)/(1.+H_ol(g,f,T_s,tau))

H_n = lambda g,f,T_s,tau: H_cl(g,f,T_s,tau)/H_wfs(s2f(f),T_s)
'''
Hwfs = lambda s, Ts: (1. - np.exp(-Ts*s))/(Ts*s)
Hzoh=Hwfs
Hlag = lambda s,tau: np.exp(-tau*s)
Hint = lambda s, Ts: 1./(1. - np.exp(-Ts*s))
Hcont = lambda s, g, Ts: g*Hint(s, Ts)
Holsplane = lambda s, Ts, tau, g:  Hwfs(s, Ts)*Hlag(s,tau)*Hcont(s, g, Ts)*Hzoh(s,Ts)
Hol = lambda f, Ts, tau, g:  Holsplane(1.j*2.*np.pi*f,Ts,tau,g)
Hrej = lambda f, Ts, tau, g: 1./(1. + Hol(f, Ts, tau, g))
Hcl = lambda f, Ts, tau, g: Hol(f, Ts, tau, g)/(1. + Hol(f, Ts, tau, g))
Hn = lambda f, Ts, tau, g: Hcl(f, Ts, tau, g)/Hwfs(1.j*2.*np.pi*f, Ts)
