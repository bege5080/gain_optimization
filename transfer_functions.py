'''
all the transfer functions and how to get their magnitude and phase
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

#magnitude and phase of transfer functions:
mag = lambda tf: 20.*np.log10(np.sqrt(np.real(tf)**2.+np.imag(tf)**2.)) #magnitude in dB
phase = lambda tf: np.unwrap(np.arctan2(np.imag(tf),np.real(tf)))*180./np.pi #np.unwrap prevents 2 pi phase discontinuities
s2f = lambda f: 1.j*2.*np.pi*f

#transfer functions
Hwfs = lambda s, Ts: (1. - np.exp(-Ts*s))/(Ts*s)
Hzoh=Hwfs
Hlag = lambda s,tau: np.exp(-tau*s)
Hint = lambda s, Ts: 1./(1. - np.exp(-Ts*s))
Hcont = lambda s, g, Ts: g*Hint(s, Ts)
Holsplane = lambda s, Ts, tau, g:  Hwfs(s, Ts)*Hlag(s,tau)*Hcont(s, g, Ts)*Hzoh(s,Ts)
Hol = lambda f, Ts, tau, g:  Holsplane(1.j*2.*np.pi*f,Ts,tau,g) #convert from s plane to frequency axis
Hrej = lambda f, Ts, tau, g: 1./(1. + Hol(f, Ts, tau, g))
Hcl = lambda f, Ts, tau, g: Hol(f, Ts, tau, g)/(1. + Hol(f, Ts, tau, g))
Hn = lambda f, Ts, tau, g: Hcl(f, Ts, tau, g)/Hwfs(1.j*2.*np.pi*f, Ts)
