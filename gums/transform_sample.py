"""
Transform sample from GUMS data
"""

import sys, os, pickle
import numpy as np, pandas as pd, h5py, scipy, scipy.stats, emcee, tqdm
from multiprocessing import Pool
from copy import deepcopy as copy
from astropy.coordinates import SkyCoord

for cardinal in ['north', 'south']:

    gums_sample = pd.read_csv(f'/data/asfe2/Projects/mwtrace_data/gums/gums_{cardinal}_b80.csv').to_dict('list')
    for key in gums_sample.keys():
        gums_sample[key] = np.array(gums_sample[key])

    # Remove any binary stars
    include = [sei[17] in [' ','V','+'] for sei in gums_sample['source_extended_id']]
    for key in gums_sample.keys():
        gums_sample[key] = gums_sample[key][include]

    # Parameter transformations
    gal = SkyCoord(ra=gums_sample['ra'], dec=gums_sample['dec'], unit='deg', frame='icrs').galactic
    gums_sample['l'] = gal.l.deg
    gums_sample['sinb'] = np.sin(gal.b.rad)

    gums_sample['s'] = gums_sample.pop('barycentric_distance')/1e3
    gums_sample['m'] = gums_sample.pop('mag_g')
    gums_sample['M'] = gums_sample['m'] - 10 - 5*np.log10(gums_sample['s'])
    gums_sample['source_id'] = gums_sample['source_id']

    import scanninglaw.asf as asf
    from scanninglaw.source import Source
    from scanninglaw.config import config
    config['data_dir'] = '/data/asfe2/Projects/testscanninglaw/'
    asf.fetch('dr3_nominal')
    dr3_asf = asf.asf(version='dr3_nominal')
    c=Source(l=gums_sample['l'], b=np.arcsin(gums_sample['sinb']), unit='rad', frame='galactic', photometry={'gaia_g':gums_sample['m']})
    gums_sample['parallax_error'] = np.sqrt(dr3_asf(c)[2,2])
    gums_sample['parallax_obs'] = np.random.normal(1/gums_sample['s'], gums_sample['parallax_error'])

    #%% Save data in HDF5 format
    filename = f'/data/asfe2/Projects/mwtrace_data/gums/gums_{cardinal}_b80.h'
    print('Saving...' + filename)
    with h5py.File(filename, 'w') as hf:
        for k in ['source_id', 's', 'sinb', 'l', 'M', 'm','parallax_error', 'parallax_obs']:
            print(k, len(gums_sample[k]))
            hf.create_dataset('sample/'+k, data=gums_sample[k], compression = 'lzf', chunks = True)
