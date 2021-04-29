import sys, os
import numpy as np, scipy, healpy as hp
from numba import njit
import numba_special

# Numba functionality
home = os.path.expanduser("~")
sys.path.append(home+'Documents/software/basecode/')

from astropy.coordinates import SkyCoord
import astropy.units as units

import selectionfunctions.cog_ii as CoGii
from selectionfunctions.std_paths import *

from selectionfunctions.carpentry import chisel
from selectionfunctions.source import Source


def apply_subgaiasf(l_sample, b_sample, G_sample, get_prob=True, dr2_sf=None, sub_sf=None, _nside=64):

    if dr2_sf is None: dr2_sf = CoGii.dr2_sf(version='modelAB', crowding=False)

    c=Source(l=l_sample, b=b_sample, unit='rad', frame='galactic', photometry={'gaia_g':G_sample})
    _eq=c.coord.icrs
    _ra=_eq.ra.deg; _dec=_eq.dec.deg

    # Collapse _n down to _nside
    _n_field = np.median(dr2_sf._n_field.reshape(hp.nside2npix(_nside), dr2_sf._n_field.shape[0]//hp.nside2npix(_nside)), axis=1).astype(int)
    pix_sample = hp.ang2pix(_nside, _ra, _dec, lonlat=True, nest=True)
    _n_sample = _n_field[pix_sample]

    _alpha_sample, _beta_sample = dr2_sf._interpolator(G_sample)
    # 0.01 < alpha,beta < 1000, make it so!
    _alpha_sample[_alpha_sample<1e-2] = 1e-2; _alpha_sample[_alpha_sample>1e+2] = 1e+2
    _beta_sample[_beta_sample<1e-2] = 1e-2; _beta_sample[_beta_sample>1e+2] = 1e+2

    sf_prob = gaia_sf(_alpha_sample, _beta_sample, _n_sample, G_sample)

    # Sub sf prob
    if sub_sf is not None:
        sf_prob *= sub_sf(c)

    subset = sf_prob>np.random.rand(len(_n_sample))

    if get_prob: return subset, sf_prob
    return subset

    # c=Source(l=l_sample, b=b_sample, unit='rad', frame='galactic')
    #
    # # Gaia SFprob:
    # _gaia_sfprob = dr2_sf(coords)
    # # Sub SF prob
    # _sub_sfprob = sub_sf(coords)
    #
    # sf_prob = _gaia_sfprob*_sub_sfprob
    # subset = sf_prob>np.random.rand(len(_n_sample))
    #
    # if get_prob: return subset, sf_prob
    # return subset

def get_subgaiasf_pars(theta=np.pi/3, nskip=2, _nside=64, dr2_sf=None, sub_sf=None, _m_grid=np.arange(0.,25.01,0.1)):

    # Get healpix coordinates
    hp_ra, hp_dec = hp.pix2ang(_nside, np.arange(hp.nside2npix(_nside)), nest=True, lonlat=True)
    hp_eq=SkyCoord(ra=hp_ra, dec=hp_dec, unit='deg')
    hp_gal=hp_eq.galactic
    hp_l=hp_gal.l.deg; hp_b=np.abs(hp_gal.b.deg)

    print('Higher Res:')
    # HEALPix at high res
    nside_highres=1024
    rapix, decpix = hp.pix2ang(nside_highres, np.arange(hp.nside2npix(nside_highres)), lonlat=True, nest=True)
    _eq=SkyCoord(ra=rapix, dec=decpix, unit='deg', frame='icrs')
    _gal=_eq.galactic; bpix_highres=np.abs(_gal.b.rad)
    pixweight = np.sum(np.reshape(bpix_highres>theta, (-1, int ( nside_highres/_nside )**2)), axis=1)/ int ( nside_highres/_nside )**2

    # Number of scans in each pixel
    _b_pixels = hp_b[pixweight>0];
    pixel_area = 4*np.pi/hp.nside2npix(_nside) * pixweight[pixweight>0]
    pixel_id = np.arange(hp.nside2npix(_nside))[pixweight>0]

    # Bin in sinb and get unique sinb bins
    _sinb_bins = np.linspace(np.sin(theta), 1, 11); _sinb_vals = (_sinb_bins[1:] + _sinb_bins[:-1])/2
    _sinb_pixels = _sinb_vals[((np.sin(np.deg2rad(_b_pixels)) - np.sin(theta) )\
                               /(1-np.sin(theta)) * 10).astype(int)]
    uni_sinb_pixels, idx_sinb_pixels = np.unique(_sinb_pixels, return_inverse=True)


    gg, ll = np.meshgrid(_m_grid, hp_l[pixel_id])
    gg, bb = np.meshgrid(_m_grid, hp_b[pixel_id])
    coords = Source(ll, bb, unit='deg', frame='galactic', photometry={'gaia_g':gg})

    print('Gaia SF:')
    # Gaia SFprob:
    #_n_pixels = np.median(dr2_sf._n_field.reshape(-1, int(4096/_nside)**2), axis=1).astype(int)[pixweight>0]
    _n_pixels = np.median(dr2_sf._n_field.reshape(hp.nside2npix(_nside), dr2_sf._n_field.shape[0]//hp.nside2npix(_nside)), axis=1).astype(int)[pixweight>0]
    _alpha, _beta = dr2_sf._alpha_interpolator(_m_grid), dr2_sf._beta_interpolator(_m_grid)
    _alpha[_alpha<1e-2] = 1e-2; _alpha[_alpha>1e+2] = 1e+2
    _beta[_beta<1e-2] = 1e-2; _beta[_beta>1e+2] = 1e+2
    _alpha,_=np.meshgrid(_alpha, _n_pixels)
    _beta,_=np.meshgrid(_beta, _n_pixels)
    _m,_n=np.meshgrid(_m_grid, _n_pixels)
    _sfprob = gaia_sf(_alpha, _beta, _n, _m)
    print('Ast SF:', gg.shape)
    # Sub SF prob
    if sub_sf is not None: _sfprob *= sub_sf(coords)


    gsf_pars={'uni_sinb_pixels':uni_sinb_pixels, 'idx_sinb_pixels':idx_sinb_pixels,
              'pixel_area':pixel_area, 'pixel_id':pixel_id,
              '_selectionfunction':_sfprob, '_m_grid':_m_grid}

    return gsf_pars


def apply_gaiasf(l_sample, b_sample, G_sample, get_prob=True, dr2_sf=None, _nside=64):

    if dr2_sf is None: dr2_sf = CoGii.dr2_sf(version='modelAB', crowding=False)

    _gal=SkyCoord(l=l_sample, b=b_sample, unit='rad', frame='galactic')
    _eq=_gal.icrs
    _ra=_eq.ra.deg; _dec=_eq.dec.deg

    # Collapse _n down to _nside
    _n_field = np.median(dr2_sf._n_field.reshape(-1, int(4096/_nside)**2), axis=1).astype(int)
    pix_sample = hp.ang2pix(_nside, _ra, _dec, lonlat=True, nest=True)
    _n_sample = _n_field[pix_sample]

    _alpha_sample, _beta_sample = dr2_sf._interpolator(G_sample)
    # 0.01 < alpha,beta < 1000, make it so!
    _alpha_sample[_alpha_sample<1e-2] = 1e-2; _alpha_sample[_alpha_sample>1e+2] = 1e+2
    _beta_sample[_beta_sample<1e-2] = 1e-2; _beta_sample[_beta_sample>1e+2] = 1e+2

    sf_prob = gaia_sf(_alpha_sample, _beta_sample, _n_sample, G_sample)
    subset = sf_prob>np.random.rand(len(_n_sample))

    if get_prob: return subset, sf_prob
    return subset

def get_gaiasf_pars(theta=np.pi/3, nskip=2, _nside=64, dr2_sf=None):

    if dr2_sf is None: dr2_sf = CoGii.dr2_sf(version='modelAB', crowding=False)

    _m_grid=dr2_sf._g_grid[::nskip]
    _alpha, _beta = dr2_sf._alpha[::nskip], dr2_sf._beta[::nskip]
    # 0.01 < alpha,beta < 1000, make it so!
    _alpha[_alpha<1e-2] = 1e-2; _alpha[_alpha>1e+2] = 1e+2
    _beta[_beta<1e-2] = 1e-2; _beta[_beta>1e+2] = 1e+2

    # Collapse _n down to 128
    _n_field = np.median(dr2_sf._n_field.reshape(-1, int(4096/_nside)**2), axis=1).astype(int)

    # Get healpix coordinates
    hp_ra, hp_dec = hp.pix2ang(_nside, np.arange(hp.nside2npix(_nside)), nest=True, lonlat=True)
    hp_eq=SkyCoord(ra=hp_ra, dec=hp_dec, unit='deg')
    hp_gal=hp_eq.galactic
    hp_l=hp_gal.l.deg; hp_b=np.abs(hp_gal.b.deg)

    # HEALPix at high res
    nside_highres=1024
    rapix, decpix = hp.pix2ang(nside_highres, np.arange(hp.nside2npix(nside_highres)), lonlat=True, nest=True)
    _eq=SkyCoord(ra=rapix, dec=decpix, unit='deg', frame='icrs')
    _gal=_eq.galactic; bpix_highres=np.abs(_gal.b.rad)
    pixweight = np.sum(np.reshape(bpix_highres>theta, (-1, int ( nside_highres/_nside )**2)), axis=1)/ int ( nside_highres/_nside )**2

    # Number of scans in each pixel
    _n_pixels = _n_field[pixweight>0]; _b_pixels = hp_b[pixweight>0];
    pixel_area = 4*np.pi/hp.nside2npix(_nside) * pixweight[pixweight>0]
    pixel_id = np.arange(hp.nside2npix(_nside))[pixweight>0]

    # Get unique scan numbers
    uni_n_pixels, idx_n_pixels = np.unique(_n_pixels, return_inverse=True)

    # Bin in sinb and get unique sinb bins
    _sinb_bins = np.linspace(np.sin(theta), 1, 11); _sinb_vals = (_sinb_bins[1:] + _sinb_bins[:-1])/2
    _sinb_pixels = _sinb_vals[((np.sin(np.deg2rad(_b_pixels)) - np.sin(theta) )\
                               /(1-np.sin(theta)) * 10).astype(int)]
    uni_sinb_pixels, idx_sinb_pixels = np.unique(_sinb_pixels, return_inverse=True)

    # Solution grid
    _uni_grid = np.zeros((len(uni_n_pixels), len(uni_sinb_pixels)))

    gsf_pars={'uni_n_pixels':uni_n_pixels, 'uni_sinb_pixels':uni_sinb_pixels,
              'idx_n_pixels':idx_n_pixels, 'idx_sinb_pixels':idx_sinb_pixels,
              'pixel_area':pixel_area, 'pixel_id':pixel_id,
              '_alpha':_alpha, '_beta':_beta, '_m_grid':_m_grid}

    return gsf_pars

#@njit
def gaia_sf(_alpha, _beta, _n, _m_grid):
    # Implementation of Gaia's selection function
    _result = np.ones(_alpha.shape)
    for _m in np.arange(1,5)[::-1]:
        _result = 1 + _result*((_n-_m+1)/_m)*(_alpha+_m-1)/(_beta+_n-_m)
    _result = 1- _result*numba_special.vec_beta(_alpha,_beta+_n)/numba_special.vec_beta(_alpha,_beta)
    _result[_m_grid>=22] = 0.
    return _result
