import sys, os, pickle, time, warnings, h5py
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, scipy, scipy.stats as stats, tqdm, h5py, emcee
from copy import deepcopy as copy
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore', r'invalid value')
warnings.filterwarnings('ignore', r'overflow encountered')
warnings.filterwarnings('ignore', r'divide by zero')

sys.path.extend(['../utilities/', '../models/', '../mockmodel/'])
import samplers, plotting, transformations, sf_utils
from transformations import func_inv_jac, func_labels, label_dict

import disk_halo_mstogap as dh_msto
# Load class instance
from TracerFit import mwfit, int_idx


if __name__=='__main__':

    times = []; checkpoints = []
    times.append(time.time()); checkpoints.append('start')

  #   # Download crossmatched catalogue
  #   b_min = 80
  #   columns = ["source_id", "l", "b", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error", "astrometric_params_solved"]
  #   columns_edr3 = ["source_id", "parallax", "parallax_error", "astrometric_params_solved", "nu_eff_used_in_astrometry", "pseudocolour"]
  #   file = f'/data/asfe2/Projects/mwtrace_data/gaia/dr2xedr3_b{b_min}.h'
  #
  #   tstart = time.time()
  #   query = f"""select * from (select {', '.join(columns)} from gaia_dr2.gaia_source
  #                               where abs(b)>{b_min} and parallax is not Null) as dr2
  # left join lateral (select {', '.join([col+' as '+col+'_edr3' for col in columns_edr3])} from gaia_edr3.gaia_source
  #                               where dr2.source_id=dr2_source_id) as edr3 on true;"""
  #   print(query)
  #   data = sqlutilpy.get(query, asDict=True, **getdata.sql_args)
  #   print(f"Time taken: {time.time()-tstart}")
  #
  #   columns += [col+'_edr3' for col in columns_edr3]

    run_id=3
    size = 100000
    file = "gaia"
    file = "gaia_edr3.gaia_source_b80"
    cardinal = "north"
    # Load Sample
    #keys = {'phot_g_mean_mag':'phot_g_mean_mag', 'parallax':'parallax', 'b':'b', 'parallax_error':'parallax_error', }
    keys = ['source_id', 'phot_g_corr', 'parallax', 'b', 'parallax_error', 'zeropoint']
    sample = {};
    filename="/data/asfe2/Projects/mwtrace_data/gaia/%s.h" % file
    with h5py.File(filename, 'r') as hf:
        subset = ( ~np.isnan(hf[cardinal]['phot_g_corr'][...]) )
        print('%d/%d' % (np.sum(subset), len(subset)))
        subsample  = np.sort(np.random.choice(np.arange(np.sum(subset)), size=size, replace=False))
        for key in keys:
            sample[key]=hf[cardinal][key][...][subset][subsample]
    sample['sinb'] = np.sin(np.deg2rad(sample['b']))
    sample['parallax_obs'] = sample['parallax']-sample['zeropoint']
    sample['m']  = sample['phot_g_corr'].copy()
    print(sample.keys())

    sample['M'] = sample['m'] + 5*(np.log10(sample['parallax_obs'])-2)
    subset = (sample['M']<12)&(sample['m']>5)&(sample['m']<22)
    print('MG<12 cut: %d/%d' % (np.sum(sample['M']<12), len(subset)))
    print('G<22 cut: %d/%d' % (np.sum(sample['m']<22), len(subset)))
    print('G>5 cut: %d/%d' % (np.sum(sample['m']>5), len(subset)))
    for key in sample.keys():
        sample[key] = sample[key][subset]


    # Get Gaia Selection Function
    from selectionfunctions.carpentry import chisel
    import selectionfunctions.cog_ii as CoGii
    from selectionfunctions.config import config
    config['data_dir'] = '/data/asfe2/Projects/testselectionfunctions/'
    #CoGii.fetch()
    dr3_sf = CoGii.dr3_sf(version='modelAB',crowding=False)

    config['data_dir'] = '/data/asfe2/Projects/astrometry/PyOutput/'
    M = 85; C = 1; jmax=4; lm=0.3; nside=32; ncores=80; B=2.0
    map_fname = f"chisquare_astrometry_jmax{jmax}_nside{nside}_M{M}_CGR{C}_lm{lm}_B{B:.1f}_ncores{ncores}_scipy_results.h5"
    ast_sf = chisel(map_fname=map_fname, nside=nside, C=C, M=M,
                    basis_options={'needlet':'chisquare', 'j':jmax, 'B':B, 'p':1.0, 'wavelet_tol':1e-2},
                    spherical_basis_directory='/data/asfe2/Projects/astrometry/SphericalWaveletsApply/')

    message = f"""\n{run_id:03d} ---> {file}, {cardinal}, MG<12, 5<G<12, Sample size: {size:d}
                 11 free parameters. hz_halo limited [3.5,7.3]. all alpha3 fixed. dirichlet alpha=2.
                 perr gradient evaluation made numerically. ftol=1e-12, gtol=1e-7. When lnp=nan in mcmc - return 1e-20.
                 Gaia SF: EDR3 (from Scanning Law)
                 Astrometry SF: {map_fname}."""
    with open(f'/data/asfe2/Projects/mwtrace_data/gaia/messages.txt', 'a') as f:
        f.write(message)
    print(message)

    # {'alpha1', 'alpha2'}
    shared_pars = {'Mms1':9, 'Mms':8, 'Mms2':7, 'Mto':3.1}
    fixed_pars = {'Mx':12, 'R0':8.27, 'theta_deg':80, 'Mms1':9, 'Mms':8, 'Mms2':7,
                  0:dict({'alpha3':-0.6}, **shared_pars),
                  1:dict({'alpha3':-0.73}, **shared_pars),
                  2:dict({'alpha3':-0.64}, **shared_pars),
                  'shd':{}}

    free_pars = {}
    free_pars[0] = ['w', 'hz', 'fD']#, 'alpha3']
    free_pars[1] = ['w', 'hz', 'fD']#, 'alpha3']
    free_pars[2] = ['w', 'hz', 'fD']#, 'alpha3']
    free_pars['shd'] = ['alpha1', 'alpha2']
    print("Free parameters: ", free_pars)

    # Transformations
    # transform, p1, p2, lower bound, upper bound
    param_trans = {}
    a_dirichlet = 2
    param_trans['shd'] = {'alpha1':('nexp',0,0,-5,3,'none'),
                          'alpha2':('nexp',0,0,-5,3,'none')}
    param_trans[0] = {'w':('exp',0,0,-10,20,'dirichlet',a_dirichlet),
                      'fD': ('logit_scaled', 0,1, -10,10,'logistic'),
                      'alpha3':('nexp',0,0,-3,3,'none'),
                      'hz': ('logit_scaled', 0.1,  0.6,-10,10,'logistic')}
    param_trans[1] = {'w':('exp',0,0,-10,20,'dirichlet',a_dirichlet),
                      'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                      'alpha3':('nexp',0,0,-3,3,'none'),
                      'hz': ('logit_scaled', 0.6,3,-10,10,'logistic')}
    param_trans[2] = {'w':('exp',0,0,-10,20,'dirichlet',a_dirichlet),
                      'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                      'alpha3':('nexp',0,0,-1,0,'none'),
                      'hz': ('logit_scaled', 3.5,  7.3,-10,10,'logistic')}

    times.append(time.time()); checkpoints.append('initialised')

    nstep_all=5000
    if True:
        save_file = f'/data/asfe2/Projects/mwtrace_data/gaia/gaia_{file}_{size:d}_{cardinal}_sf_perr_{run_id:03d}.h'
        if os.path.exists(save_file):
            raise ValueError('File %s already exists...')

        model_sf_err = mwfit(free_pars=free_pars, fixed_pars=fixed_pars, sample=sample, sf_bool=True, perr_bool=True, sub_sf=True, param_trans=param_trans)
        model_sf_err.sample['sf_subset'] = np.ones(len(sample['m'])).astype(bool)
        model_sf_err._generate_fid_pars(dr2_sf=dr3_sf, sub_sf=ast_sf, _m_grid=ast_sf.Mbins, _nside=ast_sf.nside)
        model_sf_err._generate_kwargs()
        print('bounds:\n', model_sf_err.poisson_kwargs['param_bounds'])

        # Sample from prior
        model_sf_err.mcmc_prior()
        # Optimize with BFGS
        model_sf_err.optimize_chunk(niter=10, ncores=40, label='sf_perr_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False, 'ftol':1e-12, 'gtol':1e-7})
        # Run MCMC
        model_sf_err.mcmc(ncores=40, nsteps=nstep_all, label='sf_perr_mcmc', optimize_label='sf_perr_bfgs')
        # Save results
        model_sf_err.save(save_file, fixed_pars, mode='w')

        times.append(time.time()); checkpoints.append('SF and parallax error')

    [print(f"Time {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
    print(f"Total: {times[-1]-times[0]}")
    with open(f'/data/asfe2/Projects/mwtrace_data/gaia/messages.txt', 'a') as f:
        [f.write(f"\nTime {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
