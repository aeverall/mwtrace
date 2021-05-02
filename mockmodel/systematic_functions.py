import sys, os
import numpy as np, healpy as hp, scipy, h5py

from astropy.coordinates import SkyCoord
import astropy.units as units


def apply_gerr(l_sample, sinb_sample, G_sample, n_field, _nside=64):

    c=SkyCoord(l=l_sample, b=np.arcsin(sinb_sample), unit='rad', frame='galactic')
    _eq=c.icrs; _ra=_eq.ra.deg; _dec=_eq.dec.deg
    hpix = hp.ang2pix(_nside, _ra, _dec, lonlat=True, nest=True)

    # Collapse _n down to _nside
    n_field = np.median(n_field.reshape(hp.nside2npix(_nside), n_field.shape[0]//hp.nside2npix(_nside)), axis=1).astype(int)
    n_transit = n_field[hpix]

    print('G amplitude...')
    # Model for G error
    with h5py.File('/home/asfe2/Documents/Projects/GitRepos/selectionfunctions/selectionfunctions/examples/FIRE/median_gamp_edr3.h', 'r') as hf:
        gamp_interp = scipy.interpolate.interp1d(hf['magbin'][...]+0.05, hf['med_gamp'][...],
                                    fill_value=(hf['med_gamp'][0],hf['med_gamp'][-1]), bounds_error=False)
    phot_g_error_amp = gamp_interp(G_sample)

    print('Number of observations...')
    # G-observations efficiency
    with h5py.File('/home/asfe2/Documents/Projects/GitRepos/selectionfunctions/selectionfunctions/examples/FIRE/expected_gobs_efficiency_edr3.h', 'r') as hf:
        eff_interp = scipy.interpolate.interp1d(hf['magbin'][...]+0.05, hf['mean_eff'][...],
                                    fill_value=(0,0), bounds_error=False)
    # There are 9 CCD observations per transit in Gaia.
    # The efficiency is the expected number of CCD observations which results in a G-band measurement.
    phot_g_n_obs = n_transit * (62./7.) * eff_interp(G_sample)

    print('Draw G magnitude...')
    ### Expected G uncertainty
    IG_sample = 10**(-G_sample/2.5)
    IG_error = phot_g_error_amp * IG_sample / np.sqrt(phot_g_n_obs)
    IG_obs = np.random.normal(IG_sample, IG_error)
    G_obs = -2.5*np.log10(IG_obs)

    return G_obs

def shift_z0(s, sinb, z0=0.02):

    b = np.arcsin(sinb)
    _b = np.arctan2(s*np.tan(b) - z0/np.cos(b), s)
    _sinb = np.sin(_b)

    _s = s * np.sqrt(1 + z0/s * (z0/s - 2*sinb))

    return _s, _sinb


def extinct(l, sinb, s, G):

    from dustmaps.bayestar import BayestarQuery

    b = np.rad2deg(np.arcsin(sinb))

    # Green, Schlafly, Finkbeiner et al. (2019)
    bayestar = BayestarQuery(max_samples=1)

    # Get healpix coordinates
    c = SkyCoord(l=l*units.deg, b=b*units.deg, distance=s*units.kpc, frame='galactic')

    # Apply dustmap
    Gext = G + bayestar(c, mode='random_sample') * 3.518

    return Gext
