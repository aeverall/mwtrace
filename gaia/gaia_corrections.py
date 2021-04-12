import sys, os, pickle, time, warnings
home = os.path.expanduser("~")
# Scraping data
sys.path.append(home+'/Documents/Projects/TracerDensity/zeropoint/gaiadr3_zeropoint-master/')
sys.path.append(home+'/Documents/Projects/TracerDensity/')
import h5py, numpy as np

from zero_point import zpt
zpt.load_tables()

from gmag_correction import correct_gband

data={}
# Download catalogue
columns = ["b", "phot_g_mean_mag", "astrometric_params_solved", "nu_eff_used_in_astrometry", "pseudocolour", "bp_rp"]

catalogue = "gaia_edr3.gaia_source"
b_min = 80
file = f'/data/asfe2/Projects/mwtrace_data/gaia/{catalogue}_b{b_min}.h'

for cardinal in ["north", "south"]:

    with h5py.File(file, 'r') as hf:
        for col in columns: data[col] = hf[cardinal][col][...]

    print('astrometric_params_solved: ', np.unique(data['astrometric_params_solved']))

    # Parallax zero point
    valid = (data['astrometric_params_solved']>3) & (~np.isnan(data['phot_g_mean_mag']))
    zpvals = np.zeros(len(valid))
    zpvals[valid] = zpt.get_zpt(data['phot_g_mean_mag'][valid],
                                 data['nu_eff_used_in_astrometry'][valid],
                                 data['pseudocolour'][valid],
                                 data['b'][valid],
                                 data['astrometric_params_solved'][valid])

    # G flux correction
    valid = (~np.isnan(data['phot_g_mean_mag']))&(~np.isnan(data['bp_rp']))
    gmag_corr = np.zeros(len(valid))
    gmag_corr[valid] = correct_gband(data['bp_rp'][valid], data['astrometric_params_solved'][valid], data['phot_g_mean_mag'][valid])
    gmag_corr[~valid] = data['phot_g_mean_mag'][~valid]

    with h5py.File(file, 'a') as hf:

        try: hf.create_dataset(f"{cardinal}/zeropoint", data=zpvals)
        except RuntimeError: hf[f"{cardinal}/zeropoint"][...] = zpvals

        try: hf.create_dataset(f"{cardinal}/phot_g_corr", data=gmag_corr)
        except RuntimeError: hf[f"{cardinal}/phot_g_corr"][...] = gmag_corr
