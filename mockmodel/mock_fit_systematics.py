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
import systematic_functions

import disk_halo_mstogap as dh_msto
# Load class instance
from TracerFit import mwfit, int_idx

def load_sample(filename, sid=None, size=1000):

    # Load Sample
    sample = {}; true_pars={}; latent_pars={};
    magcuts = [-100,200]
    with h5py.File(filename, 'r') as hf:
        subset = (hf['sample']['m'][...]>magcuts[0])&(hf['sample']['m'][...]<magcuts[1])
        if sid is None:
            print('low', np.sum(hf['sample']['m'][...]<magcuts[0]))
            print('high', np.sum(hf['sample']['m'][...]>magcuts[1]))
            print('%d/%d' % (np.sum(subset), len(subset)))
            subsample  = np.sort(np.random.choice(np.arange(np.sum(subset)), size=size, replace=False))
        else:
            subsample = np.intersect1d(hf['sample']['source_id'][...][subset], sid, return_indices=True)[1]

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

    return sample, true_pars, latent_pars

def sample_perr(l, sinb, s, g):

    from scanninglaw.source import Source

    dr2_asf = asf.asf(version='cogiv_2020')
    c=Source(l=l, b=np.arcsin(sinb), unit='rad', frame='galactic', photometry={'gaia_g':g})
    parallax_error = np.sqrt(dr2_asf(c)[2,2])
    parallax_obs = np.random.normal(1/s, parallax_error)

    return parallax_obs, parallax_error

def save_and_print(message, message_file):
    print(message)
    with open(message_file, 'a') as f:
        f.write(message)


if __name__=='__main__':

    times = []; checkpoints = []
    times.append(time.time()); checkpoints.append('start')

    message_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/messages_sys.txt'

    run_id=3
    size = 1000000
    file = "sample_iso"
    filename="/data/asfe2/Projects/mwtrace_data/mockmodel/%s.h" % file

    mock_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_{file}_{size:d}_sf_perr_020.h'
    with h5py.File(mock_file, 'r') as hf:
        sf_subset = hf['sf_subset'][...]
        source_id = hf['source_id'][...]
    sample, true_pars, latent_pars = load_sample(filename, sid=source_id)

    # Load in ASF
    import scanninglaw.asf as asf
    from scanninglaw.config import config
    config['data_dir'] = '/data/asfe2/Projects/testscanninglaw/'
    asf.fetch()

    # Apply Gaia Selection Function
    from selectionfunctions.carpentry import chisel
    import selectionfunctions.cog_ii as CoGii
    from selectionfunctions.config import config
    config['data_dir'] = '/data/asfe2/Projects/testselectionfunctions/'
    #CoGii.fetch()
    dr3_sf = CoGii.dr3_sf(version='modelAB',crowding=False)
    # sample['gaiasf_subset'] = sf_utils.apply_subgaiasf(sample['l'], np.arcsin(sample['sinb']), sample['m'], dr2_sf=dr3_sf)[0]

    config['data_dir'] = '/data/asfe2/Projects/astrometry/PyOutput/'
    M = 85; C = 1; jmax=4; lm=0.3; nside=32; ncores=80; B=2.0
    map_fname = f"chisquare_astrometry_jmax{jmax}_nside{nside}_M{M}_CGR{C}_lm{lm}_B{B:.1f}_ncores{ncores}_scipy_results.h5"
    ast_sf = chisel(map_fname=map_fname, nside=nside, C=C, M=M, lengthscale_m=lm, lengthscale_c=100.,
                    basis_options={'needlet':'chisquare', 'j':jmax, 'B':B, 'p':1.0, 'wavelet_tol':1e-2},
                    spherical_basis_directory='/data/asfe2/Projects/astrometry/SphericalWavelets/')
    print("SF Mbins: ", ast_sf.Mbins)
    # sample['astsf_subset'] = sf_utils.apply_subgaiasf(sample['l'], np.arcsin(sample['sinb']),
    #                                                   sample['m'], dr2_sf=dr3_sf, sub_sf=ast_sf, _nside=ast_sf.nside)[0]
    sample['astsf_subset'] = sf_subset

    message = f"""\n{run_id:03d} ---> Mock Systematics: {file}, Sample size: {size:d}, SF ast subset: {np.sum(sample['astsf_subset']):d}
                 11 free parameters. hz_halo limited [3,7.3]. all alpha3 fixed. dirichlet alpha=2.
                 perr gradient evaluation made numerically. ftol=1e-12, gtol=1e-7. When lnp=nan in mcmc - return 1e-20.
                 Selection Function: Gaia EDR3 Scanning Law Parent, Astrometry Selection Function (not finished optimizing yet).
                 Parallax error: From ASF.
                 z0=0.0208; Bayestar2019 dust; Mto=3.0 for thin disk & halo; G_err from EDR3 scanning law."""

    true_pars = true_pars
    sample = sample

    # raise KeyboardInterrupt()
    save_and_print(message, message_file)

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

    reserved={}
    for key in sample.keys():
        reserved[key] = sample[key].copy()

    nstep_all=5000
    for sys in ['z0', 'dust', 'gerr']:#'Mto',

        print(f'\n{sys}')

        save_file = f'/data/asfe2/Projects/mwtrace_data/mockmodel/mock_sys_{file}_{size:d}_{sys}_{run_id:03d}.h'
        if os.path.exists(save_file):
            raise ValueError('File %s already exists...' % save_file)

        model_sys = mwfit(free_pars=free_pars, fixed_pars=true_pars, sample=sample, sf_bool=True, perr_bool=True, sub_sf=True, param_trans=param_trans)

        if sys == 'gerr':
            # Model for G error
            model_sys.sample['m'] = systematic_functions.apply_gerr(sample['l'].copy(), sample['sinb'].copy(), sample['m'].copy(), dr3_sf._n_field)
            # print("G_std: ", np.nanstd((model_sys.sample['m']-reserved['m'])[sample['astsf_subset']]))
            save_and_print(f"G_std: {np.nanstd((model_sys.sample['m']-reserved['m'])[sample['astsf_subset']]):.5f}", message_file)
            model_sys.sample['parallax_obs'], model_sys.sample['parallax_error'] = sample_perr(model_sys.sample['l'], model_sys.sample['sinb'],
                                                                                               model_sys.sample['s'], model_sys.sample['m'])
        elif sys == 'z0':
            # Model for z0 shift
            model_sys.sample['s'], model_sys.sample['sinb'] = systematic_functions.shift_z0(model_sys.sample['s'], model_sys.sample['sinb'], z0=0.021)
            # print('z-shift: ', np.mean(model_sys.sample['s']*model_sys.sample['sinb'] - reserved['s']*reserved['sinb']))
            save_and_print(f"z-shift: {np.mean(model_sys.sample['s']*model_sys.sample['sinb'] - reserved['s']*reserved['sinb']):.5f}", message_file)
            selection = model_sys.sample['sinb']>np.sin(np.deg2rad(model_sys.fixed_pars['theta_deg']))
            for key in model_sys.sample.keys():
                model_sys.sample[key] = model_sys.sample[key][selection]
            # print(f"Removed sources: {len(model_sys.sample['s'])} / {len(reserved['s'])}, bmin={model_sys.fixed_pars['theta_deg']:.2f}")
            save_and_print(f"Removed sources: {len(model_sys.sample['s'])} / {len(reserved['s'])}, bmin={model_sys.fixed_pars['theta_deg']:.2f}", message_file)
            # Reevaluate apparent magnitude
            model_sys.sample['m'] = model_sys.sample['M'] + 5*np.log10(100*model_sys.sample['s'])
            model_sys.sample['parallax'] = 1/model_sys.sample['s']
            model_sys.sample['parallax_obs'], model_sys.sample['parallax_error'] = sample_perr(model_sys.sample['l'], model_sys.sample['sinb'],
                                                                                               model_sys.sample['s'], model_sys.sample['m'])
        elif sys == 'dust':
            print('Dust')
            model_sys.sample['m'] = systematic_functions.extinct(sample['l'].copy(), sample['sinb'].copy(), sample['s'].copy(), sample['m'].copy())
            print('Mean ext: ', np.mean((model_sys.sample['m']-reserved['m'])[sample['astsf_subset']]))
            save_and_print(f"Mean ext: {np.mean((model_sys.sample['m']-reserved['m'])[sample['astsf_subset']]):.2f}", message_file)
            model_sys.sample['parallax_obs'], model_sys.sample['parallax_error'] = sample_perr(model_sys.sample['l'], model_sys.sample['sinb'],
                                                                                               model_sys.sample['s'], model_sys.sample['m'])
        elif sys == 'Mto':
            print('Mto')
            model_sys.fixed_pars[0]['Mto'] = 3.0
            model_sys.fixed_pars[2]['Mto'] = 3.0
        else:
            raise ValueError(f'What is sys? {sys}')
        model_sys.sample['astsf_subset'] = sf_utils.apply_subgaiasf(model_sys.sample['l'], np.arcsin(model_sys.sample['sinb']),
                                                                    model_sys.sample['m'], dr2_sf=dr3_sf, sub_sf=ast_sf, _nside=ast_sf.nside)[0]
        save_and_print(f"Sample size: {np.sum(model_sys.sample['astsf_subset']):.2f}", message_file)
        model_sys.sample['sf_subset'] = sample['astsf_subset'].copy()
        model_sys._generate_fid_pars(dr2_sf=dr3_sf, sub_sf=ast_sf, _m_grid=ast_sf.Mbins, _nside=ast_sf.nside)
        model_sys._generate_kwargs()
        print('bounds:\n', model_sys.poisson_kwargs['param_bounds'])

        # Sample from prior
        model_sys.mcmc_prior()
        # Check true parameters
        true_params_f = model_sys.transform_params(model_sys.get_true_params(true_pars))
        print("True likelihood: ", model_sys.evaluate_likelihood(true_params_f))
        print(true_params_f, end="\n")
        # Optimize with BFGS
        model_sys.optimize_chunk(niter=10, ncores=40, label='sf_perr_bfgs', method='L-BFGS-B', verbose=True, minimize_options={'disp':False, 'ftol':1e-12, 'gtol':1e-7})
        # Run MCMC
        model_sys.mcmc(ncores=40, nsteps=nstep_all, label='sf_perr_mcmc', optimize_label='sf_perr_bfgs')
        # Save results
        model_sys.save(save_file, true_pars, mode='w')

        times.append(time.time()); checkpoints.append(sys)

        sample={}
        for key in reserved.keys(): sample[key] = reserved[key].copy()

    [print(f"Time {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
    print(f"Total: {times[-1]-times[0]}")
    with open(message_file, 'a') as f:
        [f.write(f"\nTime {checkpoints[i+1]}: {(times[i+1]-times[i]):.0f}s") for i in range(len(checkpoints)-1)]
