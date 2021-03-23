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

    run_id=5

    size = 99900
    file = "sample"
    # Load Sample
    sample = {}; true_pars={}; latent_pars={};
    filename="/data/asfe2/Projects/mwtrace_data/mockmodel/%s.h" % file
    with h5py.File(filename, 'r') as hf:
        subset = (hf['sample']['m'][...]>0)&(hf['sample']['m'][...]<33)
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

    # Apply Gaia Selection Function
    print(sample.keys())
    sample['gaiasf_subset'] = sf_utils.apply_gaiasf(sample['l'], np.arcsin(sample['sinb']), sample['m'])[0]

    true_pars = true_pars
    sample = sample

    message = f"""\n{run_id:03d} ---> Sample size: {size:d}, SF subset: {np.sum(sample['gaiasf_subset']):d} \n
                 fD and alpha3 free - 14 free parameters. perr gradient evaluation made numerically.
                 ftol=1e-12, gtol=1e-7. When lnp=nan in mcmc - return 1e-20."""
    with open(f'/data/asfe2/Projects/mwtrace_data/mockmodel/messages.txt', 'a') as f:
        f.write(message)
    print(message)

    free_pars = {}
    free_pars[0] = ['w', 'hz', 'fD', 'alpha3']
    free_pars[1] = ['w', 'hz', 'fD', 'alpha3']
    free_pars[2] = ['w', 'hz', 'fD', 'alpha3']
    free_pars['shd'] = ['alpha1', 'alpha2']
    print("Free parameters: ", free_pars)

    times.append(time.time()); checkpoints.append('initialised')

    if False:
        save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_full_{run_id:03d}.h'
        if os.path.exists(save_file):
            raise ValueError('File %s already exists...')

        model_full = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=False, perr_bool=False)
        model_full._generate_fid_pars()
        model_full._generate_kwargs()
        print('bounds', model_full.poisson_kwargs['param_bounds'])
        # Sample from prior
        model_full.mcmc_prior()
        # Check true parameters
        true_params_f = model_full.transform_params(model_full.get_true_params(true_pars))
        print("True likelihood: ", model_full.evaluate_likelihood(true_params_f))
        print(true_params_f, end="\n")
        # Optimize with BFGS
        model_full.optimize_parallel(niter=5, ncores=5, label='full_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False})
        # Run MCMC
        model_full.mcmc(ncores=20, nsteps=5000, label='full_mcmc', optimize_label='full_bfgs')
        # Save results
        model_full.save(save_file, true_pars, mode='w')

    times.append(time.time()); checkpoints.append('full sample')

    if False:
        save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_sf_{run_id:03d}.h'
        if os.path.exists(save_file):
            raise ValueError('File %s already exists...')

        model_sf = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=True, perr_bool=False)
        model_sf._generate_fid_pars()
        model_sf._generate_kwargs()
        print('bounds', model_sf.poisson_kwargs['param_bounds'])
        # Sample from prior
        model_sf.mcmc_prior()
        # Check true parameters
        true_params_f = model_sf.transform_params(model_sf.get_true_params(true_pars))
        print("True likelihood: ", model_sf.evaluate_likelihood(true_params_f))
        print(true_params_f, end="\n")
        # Optimize with BFGS
        model_sf.optimize_parallel(niter=5, ncores=5, label='sf_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False})
        # Run MCMC
        model_sf.mcmc(ncores=20, nsteps=5000, label='sf_mcmc', optimize_label='sf_bfgs')
        # Save results
        model_sf.save(save_file, true_pars, mode='w')

    times.append(time.time()); checkpoints.append('SF selected')

    if True:
        print('nstep:', 5000)
        save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_sf_perr_{run_id:03d}.h'
        if os.path.exists(save_file):
            raise ValueError('File %s already exists...')

        model_sf_err = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=True, perr_bool=True)
        model_sf_err._generate_fid_pars()
        model_sf_err._generate_kwargs()
        print('bounds', model_sf_err.poisson_kwargs['param_bounds'])
        # Sample from prior
        model_sf_err.mcmc_prior()
        # Check true parameters
        true_params_f = model_sf_err.transform_params(model_sf_err.get_true_params(true_pars))
        print("True likelihood: ", model_sf_err.evaluate_likelihood(true_params_f))
        print(true_params_f, end="\n")
        # Optimize with BFGS
        model_sf_err.optimize_chunk(niter=5, ncores=40, label='sf_perr_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False, 'ftol':1e-12, 'gtol':1e-7})
        # model_sf_err.optimize_results['x']['sf_perr_bfgs'] = np.array([[  9.02373448,   0.06519842,   2.87247407,   0.20566405,
        #                                              10.3484506 ,  -0.99045504,   7.80286892,  -5.        ,
        #                                              18.93727588, -10.        ,  -1.29237893,  -5.        ,
        #                                             -1.46861251,  -0.51517986]])
        # model_sf_err.optimize_results['lnp']['sf_perr_bfgs'] = np.array([206000])
        # Run MCMC
        model_sf_err.mcmc(ncores=40, nsteps=2000, label='sf_perr_mcmc', optimize_label='sf_perr_bfgs')
        # Save results
        model_sf_err.save(save_file, true_pars, mode='w')

    times.append(time.time()); checkpoints.append('SF and parallax error')
    [print(f"Time {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
    print(f"Total: {times[-1]-times[0]}")
    with open(f'/data/asfe2/Projects/mwtrace_data/mockmodel/messages.txt', 'a') as f:
        [f.write(f"\nTime {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
