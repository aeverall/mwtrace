import sys, os


import numpy as np, emcee, tqdm, corner
from multiprocessing import Pool


#%% Draw sample for toy models
def ln_edsd(params, hz=1, N=1000, theta=0, sminmax=(0, np.inf), bminmax=(-np.pi/2, np.pi/2)):

    """
    log probability - exponential vertical disk profile transformed into distance-latitude.
    """

    s, b = params
    lnp = np.log(N) + np.log(np.tan(theta)**2/(4*np.pi*hz**3)) - s*np.abs(np.sin(b))/hz + 2*np.log(s) + np.log(np.cos(b))
    try:
        if (s<sminmax[0])|(s>sminmax[1])|(b<bminmax[0])|(b>bminmax[1]): return -1e10
        if (np.abs(b)<theta): return -1e10
    except ValueError:
        lnp[(s<sminmax[0])|(s>sminmax[1])|(b<bminmax[0])|(b>bminmax[1])] = -1e10
        lnp[(np.abs(b)<theta)] = -1e10
    return lnp

def ln_powerlaw(x, gamma=2, xmin=0.1):

    """
    log probability - single power law profile with power -gamma<0 down to minimum value of xmin.
    """

    lnp = -gamma*np.log(x)
    try:
        if x<xmin: return -1e10
    except ValueError: lnp[(x<xmin)] = -1e10
    return lnp


# Magnitude models
def Mbol_plmass(M, gamma=2, a=4, xmin=0.1):

    """
    probability - absolute magnitude distribution
        - power law mass distribution (power -gamma)
        - power law mass-luminosity relation (power a)
    """

    Mmax = -2.5 * a * np.log(xmin)/np.log(10)

    norm = (gamma-1)*np.log(10)*xmin**(gamma-1)/(2.5*a)
    exponent = (gamma-1)*np.log(10)*M/(2.5*a)
    prob = norm*np.exp(exponent)
    try:
        if M>Mmax: return 1e-200
    except ValueError: prob[(M>Mmax)] = 1e-200
    return prob
