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

def load_sample(filename, cardinal, sid=None, size=1000):

    keys = ['source_id', 'phot_g_corr', 'parallax', 'b', 'parallax_error', 'zeropoint']
    sample = {};
    with h5py.File(filename, 'r') as hf:
        subset = ( ~np.isnan(hf[cardinal]['phot_g_corr'][...]) )
        print('%d/%d' % (np.sum(subset), len(subset)))
        if sid is None:
            if size=='full': subsample  = np.arange(np.sum(subset))
            else: subsample  = np.sort(np.random.choice(np.arange(np.sum(subset)), size=size, replace=False))
        else:
            subsample = np.intersect1d(hf[cardinal]['source_id'][...][subset], sid, return_indices=True)[1]
        for key in keys:
            sample[key]=hf[cardinal][key][...][subset][subsample]
    sample['sinb'] = np.sin(np.deg2rad(sample['b']))
    sample['parallax_obs'] = sample['parallax']-sample['zeropoint']
    sample['m']  = sample['phot_g_corr'].copy()
    print(sample.keys())

    sample['M'] = sample['m'] + 5*(np.log10(sample['parallax_obs'])-2)
    # subset = ((sample['M']<12)|(sample['parallax_obs']/sample['parallax_error']<1))&(sample['m']>5)&(sample['m']<22)
    subset = (sample['parallax_obs']-sample['parallax_error'] < 10**((22-sample['m'])/5)) & (sample['m']>5) & (sample['m']<22)

    print('MG<12 cut: %d/%d' % (np.sum((sample['parallax_obs']-sample['parallax_error'] < 10**((22-sample['m'])/5))), len(sample['m'])))
    print('G<22 cut: %d/%d' % (np.sum(sample['m']<22), len(sample['m'])))
    print('G>5 cut: %d/%d' % (np.sum(sample['m']>5), len(sample['m'])))
    for key in sample.keys():
        sample[key] = sample[key][subset]

    print(np.sum(sample['parallax_obs']<0))

    return sample


if __name__=='__main__':

    times = []; checkpoints = []
    times.append(time.time()); checkpoints.append('start')

    run_id=7
    size = "full" # 100000
    file = "gaia"
    file = "gaia_edr3.gaia_source_b80"
    cardinal = "north"
    # Load Sample
    filename="/data/asfe2/Projects/mwtrace_data/gaia/%s.h" % file
    #keys = {'phot_g_mean_mag':'phot_g_mean_mag', 'parallax':'parallax', 'b':'b', 'parallax_error':'parallax_error', }
    sample = load_sample(filename, cardinal, sid=None, size=size)
    #raise KeyboardInterrupt()

    # Get Gaia Selection Function
    from selectionfunctions.carpentry import chisel
    import selectionfunctions.cog_ii as CoGii
    from selectionfunctions.config import config
    config['data_dir'] = '/data/asfe2/Projects/testselectionfunctions/'
    #CoGii.fetch()
    dr3_sf = CoGii.dr3_sf(version='modelAB',crowding=False)

    config['data_dir'] = '/data/asfe2/Projects/astrometry/PyOutput/'
    M = 85; C = 1; jmax=5; lm=0.3; nside=64; ncores=88; B=2.0
    map_fname = f"chisquare_astrometry_jmax{jmax}_nside{nside}_M{M}_CGR{C}_lm{lm}_B{B:.1f}_ncores{ncores}_scipy_results.h5"
    ast_sf = chisel(map_fname=map_fname, nside=nside, C=C, M=M, lengthscale_m=lm, lengthscale_c=100.,
                    basis_options={'needlet':'chisquare', 'j':jmax, 'B':B, 'p':1.0, 'wavelet_tol':1e-2},
                    spherical_basis_directory='/data/asfe2/Projects/astrometry/SphericalWavelets/')
    print("SF Mbins: ", ast_sf.Mbins)

    message = f"""\n{run_id:03d} ---> {file}, {cardinal}, p-perr<10^((22-G)/5) or Parallax SNR<1, 5<G<12, Sample size: {len(sample['source_id']):d}
                 11 free parameters. hz_halo limited [3.,7.3]. all alpha3 fixed. dirichlet alpha=2.
                 perr gradient evaluation made numerically. ftol=1e-12, gtol=1e-7. When lnp=nan in mcmc - return 1e-20.
                 Gaia SF: EDR3 (from Scanning Law)
                 Astrometry SF: {map_fname}."""

    print(message)
    # raise KeyboardInterrupt()
    with open(f'/data/asfe2/Projects/mwtrace_data/gaia/messages.txt', 'a') as f:
        f.write(message)

    # {'alpha1', 'alpha2'}
    shared_pars = {'Mms1':9, 'Mms':8, 'Mms2':7}
    fixed_pars = {'Mx':12, 'R0':8.27, 'theta_deg':80, 'Mms1':9, 'Mms':8, 'Mms2':7,
                  0:dict({'alpha3':-0.6, 'Mto':3.1}, **shared_pars),
                  1:dict({'alpha3':-0.73, 'Mto':3.1}, **shared_pars),
                  2:dict({'alpha3':-0.64, 'Mto':3.1}, **shared_pars),
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
    param_trans[0] = {'w':('exp',0,0,-10,50,'dirichlet',a_dirichlet),
                      'fD': ('logit_scaled', 0,1, -10,10,'logistic'),
                      'alpha3':('nexp',0,0,-3,3,'none'),
                      'hz': ('logit_scaled', 0.1,  0.6,-10,10,'logistic')}
    param_trans[1] = {'w':('exp',0,0,-10,50,'dirichlet',a_dirichlet),
                      'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                      'alpha3':('nexp',0,0,-3,3,'none'),
                      'hz': ('logit_scaled', 0.6,3,-10,10,'logistic')}
    param_trans[2] = {'w':('exp',0,0,-10,50,'dirichlet',a_dirichlet),
                      'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                      'alpha3':('nexp',0,0,-1,0,'none'),
                      'hz': ('logit_scaled', 3.,  7.3,-10,10,'logistic')}

    times.append(time.time()); checkpoints.append('initialised')

    nstep_all=5000
    if False:
        Hotstart=False
        save_file = f'/data/asfe2/Projects/mwtrace_data/gaia/gaia_{file}_{size}_{cardinal}_sf_perr_{run_id:03d}.h'
        backend = emcee.backends.HDFBackend(f'/data/asfe2/Projects/mwtrace_data/gaia/gaia_{file}_{size}_{cardinal}_sf_perr_{run_id:03d}_backend.h');
        if not Hotstart:
            progress=0
            if os.path.exists(save_file):
                raise ValueError(f'File {save_file} already exists...')

            model_sf_err = mwfit(free_pars=free_pars, fixed_pars=fixed_pars, sample=sample, sf_bool=True, perr_bool=True, sub_sf=True, param_trans=param_trans)
            model_sf_err.sample['sf_subset'] = np.ones(len(sample['m'])).astype(bool)
            model_sf_err._generate_fid_pars(dr2_sf=dr3_sf, sub_sf=ast_sf, _m_grid=ast_sf.Mbins, _nside=ast_sf.nside)
            model_sf_err._generate_kwargs()
            print('bounds:\n', model_sf_err.poisson_kwargs['param_bounds'])

            # Sample from prior
            model_sf_err.mcmc_prior()
            # Optimize with BFGS
            model_sf_err.optimize_chunk(niter=10, ncores=40, label='sf_perr_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False, 'maxiter':10000, 'ftol':1e-12, 'gtol':1e-7})
            times.append(time.time()); checkpoints.append('SF and parallax error - Gradient descent')
            # Save progress
            model_sf_err.save(save_file, fixed_pars, mode='w')

        elif Hotstart:
            progress = backend.get_chain().shape[0]
            with h5py.File(save_file, 'r') as hf:
                sid = hf['source_id'][...]
            # Load in saved sample
            sample = load_sample(filename, cardinal, sid=sid)
            # Load in saved state
            model_sf_err = mwfit(free_pars=free_pars, fixed_pars=fixed_pars, sample=sample, sf_bool=True, perr_bool=True, sub_sf=True, param_trans=param_trans)
            model_sf_err.sample['sf_subset'] = np.ones(len(sample['m'])).astype(bool)
            model_sf_err.load(save_file)
            model_sf_err._generate_fid_pars(dr2_sf=dr3_sf, sub_sf=ast_sf, _m_grid=ast_sf.Mbins, _nside=ast_sf.nside)
            model_sf_err._generate_kwargs()
            print('bounds:\n', model_sf_err.poisson_kwargs['param_bounds'])

        # Run MCMC
        model_sf_err.mcmc_progress(backend=backend, ncores=40, nsteps=nstep_all, progress=progress, label='sf_perr_mcmc', optimize_label='sf_perr_bfgs')
        # Save progress
        model_sf_err.save(save_file, fixed_pars, mode='w')

        times.append(time.time()); checkpoints.append('SF and parallax error')

    if True:# Mto shifted for components
        Hotstart=False
        save_file = f'/data/asfe2/Projects/mwtrace_data/gaia/gaia_{file}_{size}_{cardinal}_Mto_{run_id:03d}.h'
        backend = emcee.backends.HDFBackend(f'/data/asfe2/Projects/mwtrace_data/gaia/gaia_{file}_{size}_{cardinal}_Mto_{run_id:03d}_backend.h');

        # {'alpha1', 'alpha2'}
        fixed_pars[0]['Mto']=2.9
        fixed_pars[1]['Mto']=3.1
        fixed_pars[2]['Mto']=3.1

        if not Hotstart:
            progress=0
            if os.path.exists(save_file):
                raise ValueError(f'File {save_file} already exists...')

            model_sf_err = mwfit(free_pars=free_pars, fixed_pars=fixed_pars, sample=sample, sf_bool=True, perr_bool=True, sub_sf=True, param_trans=param_trans)
            model_sf_err.sample['sf_subset'] = np.ones(len(sample['m'])).astype(bool)
            model_sf_err._generate_fid_pars(dr2_sf=dr3_sf, sub_sf=ast_sf, _m_grid=ast_sf.Mbins, _nside=ast_sf.nside)
            model_sf_err._generate_kwargs()
            print('bounds:\n', model_sf_err.poisson_kwargs['param_bounds'])

            # Sample from prior
            model_sf_err.mcmc_prior()
            # Optimize with BFGS
            model_sf_err.optimize_chunk(niter=10, ncores=40, label='sf_perr_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False, 'maxiter':10000, 'ftol':1e-12, 'gtol':1e-7})
            times.append(time.time()); checkpoints.append('Mto SF and parallax error - Gradient descent')
            # Save progress
            model_sf_err.save(save_file, fixed_pars, mode='w')

        elif Hotstart:
            progress = backend.get_chain().shape[0]
            with h5py.File(save_file, 'r') as hf:
                sid = hf['source_id'][...]
            # Load in saved sample
            sample = load_sample(filename, cardinal, sid=sid)
            # Load in saved state
            model_sf_err = mwfit(free_pars=free_pars, fixed_pars=fixed_pars, sample=sample, sf_bool=True, perr_bool=True, sub_sf=True, param_trans=param_trans)
            model_sf_err.sample['sf_subset'] = np.ones(len(sample['m'])).astype(bool)
            model_sf_err.load(save_file)
            model_sf_err._generate_fid_pars(dr2_sf=dr3_sf, sub_sf=ast_sf, _m_grid=ast_sf.Mbins, _nside=ast_sf.nside)
            model_sf_err._generate_kwargs()
            print('bounds:\n', model_sf_err.poisson_kwargs['param_bounds'])

        # Run MCMC
        model_sf_err.mcmc_progress(backend=backend, ncores=40, nsteps=nstep_all, progress=progress, label='sf_perr_mcmc', optimize_label='sf_perr_bfgs')
        # Save progress
        model_sf_err.save(save_file, fixed_pars, mode='w')

        times.append(time.time()); checkpoints.append('Mto SF and parallax error')


    [print(f"Time {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
    print(f"Total: {times[-1]-times[0]}")
    with open(f'/data/asfe2/Projects/mwtrace_data/gaia/messages.txt', 'a') as f:
        [f.write(f"\nTime {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
