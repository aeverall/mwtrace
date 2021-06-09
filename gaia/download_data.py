import sys, os, pickle, time, warnings
home = os.path.expanduser("~")
import numpy as np, h5py
from astropy.coordinates import SkyCoord

# Scraping data
sys.path.append(home+'/Documents/software')
sys.path.append(home+'/Documents/Research/software')
import getdata, sqlutilpy

if False:
    # Download catalogue
    catalogue = "gaia_dr2.gaia_source"
    columns = ["source_id", "l", "b", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error", "astrometric_params_solved"]
    # catalogue = "gaia_edr3.gaia_source"
    # columns += ["nu_eff_used_in_astrometry", "pseudocolour"]
    b_min = 80
    file = f'/data/asfe2/Projects/mwtrace_data/gaia/{catalogue}_b{b_min}.h'

    tstart = time.time()
    query = f"""select {', '.join(columns)} from {catalogue}
                    where abs(b)>{b_min} and parallax is not Null"""
    print(query)
    data = sqlutilpy.get(query, asDict=True, **getdata.sql_args)
    print(f"Time taken: {time.time()-tstart}")

if False:
    # Download catalogue

    file = f'/data/asfe2/Projects/mwtrace_data/gaia/{catalogue}_b{b_min}.h'

    tstart = time.time()
    query = f"""select {', '.join(columns)} from {catalogue}
                    where abs(b)>{b_min} and parallax is not Null"""
    print(query)
    data = sqlutilpy.get(query, asDict=True, **getdata.sql_args)
    print(f"Time taken: {time.time()-tstart}")

if False:
    # Download crossmatched catalogue
    b_min = 80
    columns = ["source_id", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error", "astrometric_params_solved"]
    columns_edr3 = ["source_id", "l", "b", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error", "astrometric_params_solved", "nu_eff_used_in_astrometry", "pseudocolour"]
    file = f'/data/asfe2/Projects/mwtrace_data/gaia/dr2xedr3_b{b_min}.h'

    tstart = time.time()
    query = f"""select * from (select {', '.join([col+' as '+col+'_dr2' for col in columns])} from gaia_dr2.gaia_source
                                where abs(b)>{b_min} and parallax is not Null) as dr2
  left join lateral (select {', '.join(columns_edr3)} from gaia_edr3.gaia_source
                                where dr2.source_id_dr2=dr2_source_id) as edr3 on true;"""
    print(query)
    data = sqlutilpy.get(query, asDict=True, **getdata.sql_args)
    print(f"Time taken: {time.time()-tstart}")

    columns = [col+'_dr2' for col in columns] + columns_edr3

if True:
    file = '/data/asfe2/Projects/mwtrace_data/gaia/gaia_unwise_sdss_b80.h'

    ns_eq = SkyCoord(l=[0,0],b=[90,-90], unit='deg',frame='galactic').icrs

    # Get Gaia data
    extra_cols = ['l', 'b', 'astrometric_params_solved', 'nu_eff_used_in_astrometry', 'pseudocolour', 'bp_rp']
    query = f"""select gws.*, g.* from andy_everall.gaia_unwise_sdssspec_b80 as gws
                    left join lateral (select {','.join(extra_cols)} from gaia_edr3.gaia_source as g
                                where gws.source_id=g.source_id) as g on true"""
    data = sqlutilpy.get(query, asDict=True, **getdata.sql_args)

    data['w1'] = 22.5-2.5*np.log10(data['flux_w1'])
    data['w2'] = 22.5-2.5*np.log10(data['flux_w2'])
    columns = data.keys()

    ra_poles = ns_eq.ra.rad; dec_poles = ns_eq.dec.rad
    sep_poles = np.arccos(np.sin(dec_poles)*np.sin(np.deg2rad(data['dec'][:,None])) \
                  + np.cos(dec_poles)*np.cos(np.deg2rad(data['dec'][:,None]))*\
                    np.cos(ra_poles-np.deg2rad(data['ra'][:,None])))
    data['north'] = sep_poles[:,0]*180/np.pi<=10.00001
    data['south'] = sep_poles[:,1]*180/np.pi<=10.00001


with h5py.File(file, 'w') as hf:
    #north = data['b']>0
    north = data['north']
    for col in columns:
        hf.attrs["psql_query"] = query
        try:
            hf.create_dataset(f"north/{col}", data=data[col][north])
            hf.create_dataset(f"south/{col}", data=data[col][~north])
        except TypeError:
            hf.create_dataset(f"north/{col}", data=data[col][north].astype('S20'))
            hf.create_dataset(f"south/{col}", data=data[col][~north].astype('S20'))
