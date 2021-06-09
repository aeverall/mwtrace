import sys, os, sqlutilpy, numpy as np, time, h5py, healpy as hp
home = os.path.expanduser("~")
sys.path.append(home+'/Documents/software')
import getdata
import subprocess

tstart = time.time()

gaia_cols = ['source_id', 'ra', 'dec', 'l', 'b', 'parallax', 'parallax_error', 'pmra', 'pmdec', 'phot_g_mean_mag',
             'phot_bp_rp_excess_factor', 'astrometric_params_solved', 'nu_eff_used_in_astrometry', 'pseudocolour', 'bp_rp']
sdss_cols = ['objid', 'g','r','i','z']
sdss_spec_cols = ['specobjid', 'class']
wise_cols = ['source_id as wise_sid', 'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro']
unwise_cols = ['unwise_objid', 'flux_w1', 'flux_w2']


if False: # Gaia only
    table = "gaia_b80"
    # Time:
    # 55s for rand100m sample
    # 938s for full sample
    query = f"""with gaia_xm as (select {', '.join(gaia_cols)} from gaia_edr3.gaia_source as gaia
                                    where ABS(gaia.b)>80)
            select {', '.join(gaia_cols)} from gaia_xm
                    """
if False: # Wise only
    table = "wise_b80"
    # Time:
    # 55s for rand100m sample
    # 926s for full sample
    query = f"""select ra, dec, {', '.join(wise_cols)} from wise
                                 where q3c_radial_query(ra, dec, 192.85947789,  27.12825241, 10.1)
                                 or    q3c_radial_query(ra, dec, 12.85947789,  -27.12825241, 10.1))
                    """
if False:
    table = "gaia_sdss_b80"
    # Time: 203s for rand100m
    # Time: 19998s for full
    query = f"""with gaia_xm as (select {', '.join(gaia_cols)}, gs.* from gaia_edr3.gaia_source as gaia
                                    left join lateral (select original_ext_source_id, xm_flag from gaia_edr3_aux.sdssdr13bestneighbour as gs_xm
                                                        where gaia.source_id=gs_xm.source_id) as gs on true
                                    where ABS(gaia.b)>80
                              )
            select {', '.join(gaia_cols)}, sdss.* from gaia_xm
                left join lateral
                (select {', '.join(sdss_cols)} from  sdssdr9.photoobjall as sdss
                    where sdss.objid=gaia_xm.original_ext_source_id) as sdss on true
                    """
if False: # Gaia-WISE
    table = "gaia_wise_b80"
    # Time: 1119s for rand100m sample
    # Time: 70515s and chucked out a bug
    # 958s for full sample
    query = f"""with gaia as (select {', '.join(gaia_cols)} from gaia_edr3.gaia_source as gaia
                                    where ABS(gaia.b)>80
                              )
            select {', '.join(gaia_cols)}, wise.* from gaia
                left join lateral
                (select {', '.join(wise_cols)} from  wise.main as wise
                    where q3c_join_pm (gaia.ra, gaia.dec, gaia.pmra, gaia.pmdec, 1,
            		                   2016.5,  wise.ra, wise.dec, 2010, 30, 2./3600)) as wise on true
                    """
    # # Time: 71313 s
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

if True: # Gaia-unWISE
    table = "gaia_unwise_b80"
    # Time: 1119s for rand100m sample
    # 958s for full sample
    query = f"""with gaia as (select {', '.join(gaia_cols)} from gaia_edr3.gaia_source as gaia
                                    where ABS(gaia.b)>80
                              )
            select {', '.join(gaia_cols)}, wise.* from gaia
                left join lateral
                (select {', '.join(unwise_cols)} from  unwise_1901.main as wise
                    where q3c_join_pm (gaia.ra, gaia.dec, gaia.pmra, gaia.pmdec, 1,
            		                   2016.,  wise.ra, wise.dec, 2010, 30, 2./3600)
                    ORDER BY
                        q3c_dist(gaia.ra,gaia.dec,wise.ra,wise.dec)
                    ASC LIMIT 1) as wise on true
                    """

if True: # Gaia-unWISE-SDSS
    table = "gaia_unwise_sdssspec_b80"
    query = f"""with gw as (select * from andy_everall.gaia_unwise_b80 as gaia)
            select gw.*, sdss.* from gw
                left join lateral
                (select {', '.join(sdss_spec_cols)} from  sdssdr14.specobjall as sdss
                    where q3c_join_pm (gw.ra, gw.dec, gw.pmra, gw.pmdec, 1,
                                		2016.,  sdss.ra, sdss.dec, 2000, 30, 2./3600)
                    ORDER BY q3c_dist(gw.ra,gw.dec,sdss.ra,sdss.dec)
                    ASC LIMIT 1) as sdss on true
                    """

if False: # Gaia-unWISE-SDSS
    table = "gaia_unwise_sdss_b80"
    query = f"""with gw as (select * from andy_everall.gaia_unwise_b80 as gaia)
            select gw.*, sdss.* from gw
                left join lateral
                (select {', '.join(sdss_cols)} from  sdssdr14.photoobjall as sdss
                    where q3c_join_pm (gw.ra, gw.dec, gw.pmra, gw.pmdec, 1,
                                		2016.5,  sdss.ra, sdss.dec, 2000, 30, 2./3600)
                    ORDER BY q3c_dist(gw.ra,gw.dec,sdss.ra,sdss.dec)
                    ASC LIMIT 1) as sdss on true
                    """

if True:

    query = f"""DROP TABLE IF EXISTS andy_everall.{table}; set statement_timeout to 86400000; show statement_timeout;
                create table {table} as {query}"""
    print(query)

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
