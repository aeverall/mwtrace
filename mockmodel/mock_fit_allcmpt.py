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

sys.path.extend(['../utilities/', '../models/'])
import samplers, plotting, transformations, sf_utils
from transformations import func_inv_jac, func_labels, label_dict

import disk_halo_mstogap as dh_msto
# Load class instance
from TracerFit import mwfit, int_idx


if __name__=='__main__':

    times = []; checkpoints = []
    times.append(time.time()); checkpoints.append('start')

    run_id=23
    size = 1000000
    file = "sample_dr3asf"
    # Load Sample
    sample = {}; true_pars={}; latent_pars={};
    magcuts = [-100,200]
    filename="/data/asfe2/Projects/mwtrace_data/mockmodel/%s.h" % file
    with h5py.File(filename, 'r') as hf:
        print('low', np.sum(hf['sample']['m'][...]<magcuts[0]))
        print('high', np.sum(hf['sample']['m'][...]>magcuts[1]))
        subset = (hf['sample']['m'][...]>magcuts[0])&(hf['sample']['m'][...]<magcuts[1])
        print('%d/%d' % (np.sum(subset), len(subset)))
        subsample  = np.sort(np.random.choice(np.arange(np.sum(subset)), size=size, replace=False))
        for key in hf['sample'].keys():
            sample[key]=hf['sample'][key][...][subset][subsample]
        # Get true parameters
        for key in hf['true_pars'].keys():
            if not key in np.arange(3).astype(str):
                true_pars[int_idx(key)]=hf['true_pars'][key][...]
            else:
                true_pars[int_idx(key)]={}
                for par in hf['true_pars'][key].keys():
                    true_pars[int_idx(key)][par]=hf['true_pars'][key][par][...]
    for j in range(3): true_pars[j]['w']*=size
    print(sample.keys())

    if True:
        # Apply Gaia Selection Function
        from selectionfunctions.carpentry import chisel
        import selectionfunctions.cog_ii as CoGii
        from selectionfunctions.config import config
        config['data_dir'] = '/data/asfe2/Projects/testselectionfunctions/'
        #CoGii.fetch()
        dr3_sf = CoGii.dr3_sf(version='modelAB',crowding=False)
        sample['gaiasf_subset'] = sf_utils.apply_subgaiasf(sample['l'], np.arcsin(sample['sinb']), sample['m'], dr2_sf=dr3_sf)[0]

        config['data_dir'] = '/data/asfe2/Projects/astrometry/PyOutput/'
        M = 85; C = 1; jmax=5; lm=0.3; nside=64; ncores=88; B=2.0
        map_fname = f"chisquare_astrometry_jmax{jmax}_nside{nside}_M{M}_CGR{C}_lm{lm}_B{B:.1f}_ncores{ncores}_scipy_results.h5"
        ast_sf = chisel(map_fname=map_fname, nside=nside, C=C, M=M, lengthscale_m=lm, lengthscale_c=100.,
                        basis_options={'needlet':'chisquare', 'j':jmax, 'B':B, 'p':1.0, 'wavelet_tol':1e-2},
                        spherical_basis_directory='/data/asfe2/Projects/astrometry/SphericalWavelets/')
        print("SF Mbins: ", ast_sf.Mbins)
        sample['astsf_subset'] = sf_utils.apply_subgaiasf(sample['l'], np.arcsin(sample['sinb']),
                                                          sample['m'], dr2_sf=dr3_sf, sub_sf=ast_sf, _nside=ast_sf.nside)[0]
        # Set to zero as no astrometry error model beyond G=21.45
        sample['astsf_subset'][sample['m']>21.45] == 0.

        message = f"""\n{run_id:03d} ---> {file}, Sample size: {size:d}, SF subset: {np.sum(sample['gaiasf_subset']):d}, SF ast subset: {np.sum(sample['astsf_subset']):d}
                     11 free parameters. hz_halo limited [3.,7.3]. logw [0,30]. all alpha3 fixed. dirichlet alpha=2.
                     perr gradient evaluation made numerically. ftol=1e-12, gtol=1e-7. When lnp=nan in mcmc - return 1e-20.
                     Selection Function: Gaia EDR3 Scanning Law Parent, Astrometry Selection Function nside64,jmax5 - grid method.
                     Parallax error: From ASF.
                     Testing mask code."""

    true_pars = true_pars
    sample = sample

    print("Nan parallax error: ", np.sum(np.isnan(sample['parallax_error'])))
    print("Nan perr in subset: ", np.sum(sample['astsf_subset']&np.isnan(sample['parallax_error'])))

    with open(f'/data/asfe2/Projects/mwtrace_data/mockmodel/messages.txt', 'a') as f:
        f.write(message)
    print(message)

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
    param_trans[0] = {'w':('exp',0,0,0,30,'dirichlet',a_dirichlet),
                      'fD': ('logit_scaled', 0,1, -10,10,'logistic'),
                      'alpha3':('nexp',0,0,-3,3,'none'),
                      'hz': ('logit_scaled', 0.1,  0.6,-10,10,'logistic')}
    param_trans[1] = {'w':('exp',0,0,0,30,'dirichlet',a_dirichlet),
                      'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                      'alpha3':('nexp',0,0,-3,3,'none'),
                      'hz': ('logit_scaled', 0.6,3,-10,10,'logistic')}
    param_trans[2] = {'w':('exp',0,0,0,30,'dirichlet',a_dirichlet),
                      'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                      'alpha3':('nexp',0,0,-1,0,'none'),
                      'hz': ('logit_scaled', 3.,  7.3,-10,10,'logistic')}

    times.append(time.time()); checkpoints.append('initialised')

    nstep_all=5000
    if True:
        save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_sf_perr_{run_id:03d}.h'
        if os.path.exists(save_file):
            raise ValueError('File %s already exists...')

        model_sf_err = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=True, perr_bool=True, sub_sf=True, param_trans=param_trans)
        model_sf_err.sample['sf_subset'] = sample['astsf_subset'].copy()
        #model_sf_err.sample['sf_subset'] = sample['gaiasf_subset'].copy()
        model_sf_err._generate_fid_pars(dr2_sf=dr3_sf, sub_sf=ast_sf, _m_grid=ast_sf.Mbins, _nside=ast_sf.nside)
        model_sf_err._generate_kwargs()
        print('bounds:\n', model_sf_err.poisson_kwargs['param_bounds'])

        # Sample from prior
        model_sf_err.mcmc_prior()
        # Check true parameters
        true_params_f = model_sf_err.transform_params(model_sf_err.get_true_params(true_pars))
        print("True likelihood: ", model_sf_err.evaluate_likelihood(true_params_f))
        print(true_params_f, end="\n")
        # Optimize with BFGS
        model_sf_err.optimize_chunk(niter=10, ncores=40, label='sf_perr_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False, 'ftol':1e-12, 'gtol':1e-7})
        # Run MCMC
        model_sf_err.mcmc(ncores=40, nsteps=nstep_all, label='sf_perr_mcmc', optimize_label='sf_perr_bfgs')
        # Save results
        model_sf_err.save(save_file, true_pars, mode='w')

        times.append(time.time()); checkpoints.append('SF and parallax error')

    if True:
        save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_sfast_{run_id:03d}.h'
        if os.path.exists(save_file):
            raise ValueError('File %s already exists...')

        model_sf = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=True, perr_bool=False, sub_sf=True, param_trans=param_trans)
        model_sf.sample['sf_subset'] = sample['astsf_subset'].copy()
        model_sf._generate_fid_pars(dr2_sf=dr3_sf, sub_sf=ast_sf, _m_grid=ast_sf.Mbins, _nside=ast_sf.nside)
        model_sf._generate_kwargs()
        print('bounds:\n', model_sf.poisson_kwargs['param_bounds'])

        # Sample from prior
        model_sf.mcmc_prior()
        # Check true parameters
        true_params_f = model_sf.transform_params(model_sf.get_true_params(true_pars))
        print("True likelihood: ", model_sf.evaluate_likelihood(true_params_f))
        print(true_params_f, end="\n")
        # Optimize with BFGS
        model_sf.optimize_parallel(niter=10, ncores=10, label='sf_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False})
        # Run MCMC
        model_sf.mcmc(ncores=40, nsteps=nstep_all, label='sf_mcmc', optimize_label='sf_bfgs')
        # Save results
        model_sf.save(save_file, true_pars, mode='w')

        times.append(time.time()); checkpoints.append('Astrometry SF selected')

    if True:
        save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_full_{run_id:03d}.h'
        if os.path.exists(save_file):
            raise ValueError('File %s already exists...')

        model_full = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=False, perr_bool=False, param_trans=param_trans)
        model_full._generate_fid_pars()
        model_full._generate_kwargs()
        print('bounds:\n', model_full.poisson_kwargs['param_bounds'])
        # Sample from prior
        model_full.mcmc_prior()
        # Check true parameters
        true_params_f = model_full.transform_params(model_full.get_true_params(true_pars))
        print("True likelihood: ", model_full.evaluate_likelihood(true_params_f))
        print(true_params_f, end="\n")
        # Optimize with BFGS
        model_full.optimize_parallel(niter=10, ncores=10, label='full_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False})
        # Run MCMC
        model_full.mcmc(ncores=20, nsteps=nstep_all, label='full_mcmc', optimize_label='full_bfgs')
        # Save results
        model_full.save(save_file, true_pars, mode='w')

        times.append(time.time()); checkpoints.append('full sample')

    if False:
        save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_sf_perrHOT_{run_id:03d}.h'
        if os.path.exists(save_file):
            raise ValueError('File %s already exists...')

        model_sf_err = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=True, perr_bool=True, sub_sf=True, param_trans=param_trans)
        model_sf_err.sample['sf_subset'] = sample['astsf_subset'].copy()
        model_sf_err._generate_fid_pars(dr2_sf=dr3_sf, sub_sf=ast_sf, _m_grid=ast_sf.Mbins, _nside=ast_sf.nside)
        model_sf_err._generate_kwargs()
        print('bounds:\n', model_sf_err.poisson_kwargs['param_bounds'])
        # Check true parameters
        true_params_f = model_sf_err.transform_params(model_sf_err.get_true_params(true_pars))
        print("True likelihood: ", model_sf_err.evaluate_likelihood(true_params_f))
        print(true_params_f, end="\n")
        # Run MCMC
        model_sf_err.mcmc(ncores=40, p0=true_params_f, nsteps=1000, label='sf_perr_mcmc', optimize_label='sf_perr_bfgs')
        # Save results
        model_sf_err.save(save_file, true_pars, mode='w')

        times.append(time.time()); checkpoints.append('SF and parallax error HOT')

    # if False: ### Original SF method
    #     save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_sf_{run_id:03d}.h'
    #     if os.path.exists(save_file):
    #         raise ValueError('File %s already exists...')
    #
    #     model_sf = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=True, perr_bool=False, param_trans=param_trans)
    #     model_sf.sample['sf_subset'] = sample['gaiasf_subset'].copy()
    #     model_sf._generate_fid_pars(dr2_sf=dr3_sf)
    #     model_sf._generate_kwargs()
    #     print('bounds:\n', model_sf.poisson_kwargs['param_bounds'])
    #     # Sample from prior
    #     model_sf.mcmc_prior()
    #     # Check true parameters
    #     true_params_f = model_sf.transform_params(model_sf.get_true_params(true_pars))
    #     print("True likelihood: ", model_sf.evaluate_likelihood(true_params_f))
    #     print(true_params_f, end="\n")
    #     # Optimize with BFGS
    #     model_sf.optimize_parallel(niter=5, ncores=5, label='sf_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False})
    #     # Run MCMC
    #     model_sf.mcmc(ncores=20, nsteps=nstep_all, label='sf_mcmc', optimize_label='sf_bfgs')
    #     # Save results
    #     model_sf.save(save_file, true_pars, mode='w')
    #
    #     times.append(time.time()); checkpoints.append('SF selected')
    #
    # if False: ### New SF method
    #     save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_sfsub_{run_id:03d}.h'
    #     if os.path.exists(save_file):
    #         raise ValueError('File %s already exists...')
    #
    #     model_sf = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=True, perr_bool=False, sub_sf=True, param_trans=param_trans)
    #     model_sf.sample['sf_subset'] = sample['gaiasf_subset'].copy()
    #     model_sf._generate_fid_pars(dr2_sf=dr3_sf, _m_grid=np.arange(0., 22.1, 0.1))
    #     model_sf._generate_kwargs()
    #     print('bounds:\n', model_sf.poisson_kwargs['param_bounds'])
    #
    #     # Sample from prior
    #     model_sf.mcmc_prior()
    #     # Check true parameters
    #     true_params_f = model_sf.transform_params(model_sf.get_true_params(true_pars))
    #     print("True likelihood: ", model_sf.evaluate_likelihood(true_params_f))
    #     print(true_params_f, end="\n")
    #     # Optimize with BFGS
    #     model_sf.optimize_parallel(niter=5, ncores=5, label='sf_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False})
    #     # Run MCMC
    #     model_sf.mcmc(ncores=20, nsteps=nstep_all, label='sf_mcmc', optimize_label='sf_bfgs')
    #     # Save results
    #     model_sf.save(save_file, true_pars, mode='w')
    #
    #     times.append(time.time()); checkpoints.append('SF selected')

    [print(f"Time {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
    print(f"Total: {times[-1]-times[0]}")
    with open(f'/data/asfe2/Projects/mwtrace_data/mockmodel/messages.txt', 'a') as f:
        [f.write(f"\nTime {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
