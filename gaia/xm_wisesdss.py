import sys, os, sqlutilpy, numpy as np, time, h5py, healpy as hp
home = os.path.expanduser("~")
sys.path.append(home+'/Documents/software')
import getdata
import subprocess

tstart = time.time()

gaia_cols = ['source_id', 'ra', 'dec', 'parallax', 'parallax_error', 'pmra', 'pmdec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']
sdss_cols = ['objid', 'g','r','i','z']
wise_cols = ['source_id as wise_sid', 'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro']


if True: # Gaia only
    table = "gaia_b80"
    # Time:
    # 55s for rand100m sample
    # 938s for full sample
    query = f"""with gaia_xm as (select {', '.join(gaia_cols)} from gaia_edr3.gaia_source as gaia
                                    where ABS(gaia.b)>80)
            select {', '.join(gaia_cols)} from gaia_xm
                    """
if True:
    table = "gaia3_sdss_b80"
    # Time: 218s for rand100m
    # Time: 19998s for full
    query = f"""with gaia_xm as (select {', '.join(gaia_cols)}, gs.* from andy_everall.gaia3_rand100m as gaia
                                    left join lateral (select original_ext_source_id, xm_flag from gaia_edr3_aux.sdssdr13bestneighbour as gs_xm
                                                        where gaia.source_id=gs_xm.source_id) as gs on true
                                    where ABS(gaia.b)>80
                              )
            select {', '.join(gaia_cols)}, sdss.* from gaia_xm
                left join lateral
                (select {', '.join(sdss_cols)} from  sdssdr9.photoobjall as sdss
                    where sdss.objid=gaia_xm.original_ext_source_id) as sdss on true
                    """
if True:
    table = "gaia_wise_b80"
    # Time: 1119s for rand100m sample
    query = f"""with gaia as (select {', '.join(gaia_cols)} from gaia_edr3.gaia_source as gaia
                                    where ABS(gaia.b)>80
                              )
            select {', '.join(gaia_cols)}, wise.* from gaia
                left join lateral
                (select {', '.join(wise_cols)} from  wise.main as wise
                    where q3c_join_pm (gaia.ra, gaia.dec, gaia.pmra, gaia.pmdec, 1,
            		                   2016.5,  wise.ra, wise.dec, 2010, 30, 5./3600)) as wise on true
                    """
    # # Time:
    # query = f"""with gaia as (select {', '.join(gaia_cols)} from andy_everall.gaia3_rand100m as gaia
    #                                 where ABS(gaia.b)>80),
    #                  wise as (select ra, dec, {', '.join(wise_cols)} from  wise.main as wise
    #                             where q3c_radial_query(ra, dec, 192.85947789,  27.12825241, 10.1)
    #                             or    q3c_radial_query(ra, dec, 12.85947789,  -27.12825241, 10.1))
    #         select gaia.*, w_g.* from gaia
    #             left join lateral
    #             (select ra as ra_wise, dec as dec_wise, {', '.join(wise_cols)} from wise
    #                 where q3c_join_pm (gaia.ra, gaia.dec, gaia.pmra, gaia.pmdec, 1, 2016.5,
    #                                    wise.ra, wise.dec, 2010, 30, 5./3600)) as w_g on true
    #                 """


if True:

    print(query)

    query = f"""DROP TABLE IF EXISTS andy_everall.{table}; set statement_timeout to 86400000; show statement_timeout;
                create table {table} as {query}"""

    print('Running Query:')
    subprocess.call('echo "%s" | psql'  % query,shell=True)
    tq1 = time.time(); print('Query took %d s' % (tq1-tstart))

if False:
    print('Downloading data:')
    querygetdata = f"""select * from andy_everall.{table}"""
    result = sqlutilpy.get(querygetdata, asDict=True, **getdata.sql_args)
    tq2 = time.time(); print('Download took %d s' % (tq2-tq1))

    filename =f'/data/asfe2/Projects/mwtrace_data/gaia/{table}.h'
    print('saving: ', filename)
    with h5py.File(filename, 'w') as hf:
        for key in result.keys():
            hf.create_dataset(key, data=result[key])
