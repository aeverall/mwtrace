import sys, os, pickle, time, warnings
home = os.path.expanduser("~")
import numpy as np, h5py

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

if True:
    # Download crossmatched catalogue
    b_min = 80
    columns = ["source_id", "l", "b", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error", "astrometric_params_solved"]
    columns_edr3 = ["source_id", "parallax", "parallax_error", "astrometric_params_solved", "nu_eff_used_in_astrometry", "pseudocolour"]
    file = f'/data/asfe2/Projects/mwtrace_data/gaia/dr2xedr3_b{b_min}.h'

    tstart = time.time()
    query = f"""select * from (select {', '.join(columns)} from gaia_dr2.gaia_source
                                where abs(b)>{b_min} and parallax is not Null) as dr2
  left join lateral (select {', '.join([col+' as '+col+'_edr3' for col in columns_edr3])} from gaia_edr3.gaia_source
                                where dr2.source_id=dr2_source_id) as edr3 on true;"""
    print(query)
    data = sqlutilpy.get(query, asDict=True, **getdata.sql_args)
    print(f"Time taken: {time.time()-tstart}")

    columns += [col+'_edr3' for col in columns_edr3]


with h5py.File(file, 'w') as hf:
    north = data['b']>0
    for col in columns:
        hf.attrs["psql_query"] = query
        hf.create_dataset(f"north/{col}", data=data[col][north])
        hf.create_dataset(f"south/{col}", data=data[col][~north])
