import sys, os, pickle, time, warnings

import numpy as np, pandas as pd, scipy, scipy.stats as stats, tqdm, h5py, emcee
from scipy.optimize import minimize
from copy import deepcopy as copy

sys.path.extend(['../utilities/', '../models/'])
import samplers, disk_cone_plcut as dcp, plotting, transformations, sf_utils
import disk_halo_mstogap as dh_msto
from transformations import func_inv_jac


# transform, p1, p2, lower bound, upper bound
param_trans = {}
a_dirichlet = 2
param_trans['shd'] = {'alpha1':('nexp',0,0,-3,3,'none'),
                      'alpha2':('nexp',0,0,-3,3,'none')}
param_trans[0] = {'w':('exp',0,0,-10,10,'dirichlet',a_dirichlet),
                  'fD': ('logit_scaled', 0,1, -10,10,'logistic'),
                  'alpha3':('nexp',0,0,-10,10,'none'),
                  'hz': ('logit_scaled', 0.1,  1.2,-10,10,'logistic')}
param_trans[1] = {'w':('exp',0,0,-10,10,'dirichlet',a_dirichlet),
                  'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                  'alpha3':('nexp',0,0,-10,10,'none'),
                  'hz': ('logit_scaled', 1.2,3,-10,10,'logistic')}
param_trans[2] = {'w':('exp',0,0,-10,10,'dirichlet',a_dirichlet),
                  'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                  'alpha3':('nexp',0,0,-10,10,'none'),
                  'hz': ('logit_scaled', 3,  7.3,-10,10,'logistic')}


def poisson_like(params, bounds=None, grad=False):

    poisson_kwargs = copy(poisson_kwargs_global)

    # Prior boundaries
    if bounds is None: bounds = poisson_kwargs['param_bounds']
    if np.sum((params<=bounds[0])|(params>=bounds[1]))>0:
        if grad: return -1e20, np.zeros(len(params))
        else: return -1e20

    # Optional prior inclusion
    if poisson_kwargs_global['model_prior'] is not None:
        prior=poisson_kwargs_global['model_prior'](params, fid_pars=poisson_kwargs['fid_pars'], grad=grad, bounds=bounds)
    else:
        if not grad: prior=0.
        else: prior=(0.,0.)

    integral = poisson_kwargs['model_integrate'](params, bins=poisson_kwargs['bins'], fid_pars=poisson_kwargs['fid_pars'], grad=grad)
    obj = poisson_kwargs['logmodel'](poisson_kwargs['sample'], params, gmm=poisson_kwargs['gmm'], fid_pars=poisson_kwargs['fid_pars'], grad=grad)
    if not grad: return np.sum(obj) - integral + prior

    elif grad:
        model_val = np.sum(obj[0]) - integral[0] + prior[0]
        model_grad = np.sum(obj[1], axis=1) - integral[1] + prior[1]
        return model_val, model_grad

global params_i

def poisson_like_parallel_func(ii):
    grad = poisson_kwargs_global['grad']
    output = poisson_kwargs_global['logmodel'](poisson_kwargs_global['sample'][:,ii:ii+poisson_kwargs_global['chunksize']].copy(),
                                            params_i,
                                            gmm=copy(poisson_kwargs_global['gmm']),
                                            fid_pars=copy(poisson_kwargs_global['fid_pars']),
                                            grad=grad)
    if not grad: return np.sum(obj)
    else: return np.sum(output[0]), np.sum(output[1], axis=1)

def poisson_like_parallel(params, bounds=None):

    global params_i
    params_i = params.copy()
    grad = poisson_kwargs_global['grad']

    poisson_kwargs = copy(poisson_kwargs_global)

    # Prior boundaries
    if bounds is None: bounds = poisson_kwargs['param_bounds']
    if np.sum((params<=bounds[0])|(params>=bounds[1]))>0:
        if grad: return -1e20, np.zeros(len(params))
        else: return -1e20

    # Optional prior inclusion
    if poisson_kwargs_global['model_prior'] is not None:
        prior=poisson_kwargs_global['model_prior'](params, fid_pars=poisson_kwargs['fid_pars'], grad=grad, bounds=bounds)
    else:
        if not grad: prior=0.
        else: prior=(0.,0.)

    logl_val = 0.
    if grad: logl_grad = np.zeros(len(params))

    integral = poisson_kwargs['model_integrate'](params, bins=poisson_kwargs['bins'],
                                                 fid_pars=poisson_kwargs['fid_pars'], grad=grad)

    i_list = np.arange(0, poisson_kwargs_global['sample'].shape[1], poisson_kwargs_global['chunksize'])
    with Pool(poisson_kwargs['ncores']) as p:
        for logl_i in p.imap(poisson_like_parallel_func, i_list):
            if not grad: logl_val += logl_i
            if grad:
                logl_val += logl_i[0]
                logl_grad += logl_i[1]

    if not grad: return logl_val - integral[0] + prior[0]
    return logl_val - integral[0] + prior[0], logl_grad - integral[1] + prior[1]

def nloglikelihood(x):
    # Negative log likelihood and gradient for Newton-CG
    lnl, grad = poisson_like(x, grad=True)
    return -lnl, -grad

##Print callback function
def printx(Xi):
    global Nfeval
    global fout
    sys.stdout.write('At iterate {0}, {1}'.format(Nfeval, poisson_like(Xi)) + '\r')
    #sys.stdout.write('\r'+str(Xi)+'\n')
    Nfeval += 1

def save_hdf5_recurrent(obj, file, path, hf):

    for key, item in obj.items():
        if isinstance(item, dict): save_hdf5_recurrent(item, file, os.path.join(path, str(key)), hf)
        else:
            try: hf.create_dataset(os.path.join(path, str(key)), data=item)
            except TypeError: hf.create_dataset(os.path.join(path, str(key)), data=np.array(item).astype('S20'))

def save_hdf5(chain_dict, filename):
    # Save all chains
    print('Saving...' + filename)
    if os.path.exists(filename): mode='a'
    else: mode='w'
    print('Mode: %s' % mode)

    with h5py.File(filename, mode) as hf:
        save_hdf5_recurrent(chain_dict, filename, "", hf)

if __name__=='__main__':

    savefile = "/data/asfe2/Projects/mwtrace_data/mockmodel/fits_allcmpt_ncg_gaiasf.h"
    if os.path.exists(savefile): raise ValueError('File: %s already exists!' % savefile)

    nsteps=100; ncores=2;
    n_newtoncg=10

    # output dictionary:
    output = {}

    # Load Sample
    size = 5000
    sample = {}; true_pars={}; latent_pars={};
    filename = '/data/asfe2/Projects/mwtrace_data/mockmodel/sample.h'
    with h5py.File(filename, 'r') as hf:
        subset = (hf['sample']['m'][...]>0)&(hf['sample']['m'][...]<33)
        print('%d/%d' % (np.sum(subset), len(subset)))
        subsample  = np.sort(np.random.choice(np.arange(np.sum(subset)), size=size, replace=False))
        for key in hf['sample'].keys():
            sample[key]=hf['sample'][key][...][subset][subsample]
        # Get true parameters
        for key in hf['true_pars'].keys():
            if not key in np.arange(3).astype(str):
                true_pars[key]=hf['true_pars'][key][...]
            else:
                true_pars[key]={}
                for par in hf['true_pars'][key].keys():
                    true_pars[key][par]=hf['true_pars'][key][par][...]
    for j in range(3): true_pars[str(j)]['w']*=size

    # Apply Gaia Selection Function
    print(sample.keys())
    sample['gaiasf_subset'] = sf_utils.apply_gaiasf(sample['l'], np.arcsin(sample['sinb']), sample['m'])[0]


    fid_pars = {'Mmax':true_pars['Mx'],  'lat_min':np.deg2rad(true_pars['theta_deg']), 'R0':true_pars['R0'],
                'free_pars':{}, 'fixed_pars':{}, 'functions':{}, 'functions_inv':{}, 'jacobians':{}, 'w':True,
                'components':['disk','disk','halo'], 'ncomponents':3}

    fid_pars['free_pars'][0] = ['w', 'hz']
    fid_pars['free_pars'][1] = ['w', 'hz']
    fid_pars['free_pars'][2] = ['w', 'hz']
    fid_pars['free_pars']['shd'] = ['alpha1', 'alpha2']

    fid_pars['fixed_pars'][0] = {'Mms':true_pars['Mms'], 'fD':true_pars['0']['fD'], 'alpha3':true_pars['0']['alpha3'],
                                 'Mms1':true_pars['Mms1'], 'Mms2':true_pars['Mms2'],
                                 'Mto':true_pars['0']['Mto']}
    fid_pars['fixed_pars'][1] = copy(fid_pars['fixed_pars'][0]); fid_pars['fixed_pars'][2] = copy(fid_pars['fixed_pars'][0])
    fid_pars['fixed_pars'][1]['Mto'] = true_pars['1']['Mto']
    fid_pars['fixed_pars'][1]['fD'] = true_pars['1']['fD']
    fid_pars['fixed_pars'][2]['Mto'] = true_pars['2']['Mto']
    fid_pars['fixed_pars'][2]['fD'] = true_pars['2']['fD']

    fid_pars['functions']={}; fid_pars['functions_inv']={}; fid_pars['jacobians']={}; bounds=[]
    params_i = 0
    for cmpt in np.arange(fid_pars['ncomponents']).tolist()+['shd',]:
        fid_pars['functions'][cmpt]={}; fid_pars['functions_inv'][cmpt]={}; fid_pars['jacobians'][cmpt]={}
        for par in fid_pars['free_pars'][cmpt]:
            fid_pars['functions'][cmpt][par], \
            fid_pars['functions_inv'][cmpt][par], \
            fid_pars['jacobians'][cmpt][par]=func_inv_jac[param_trans[cmpt][par][0]](*param_trans[cmpt][par][1:3])
            bounds.append([param_trans[cmpt][par][3], param_trans[cmpt][par][4]])
            params_i += 1;
    bounds = np.array(bounds).T

    fid_pars['priors'] = {}
    params_i = 0
    for cmpt in np.arange(fid_pars['ncomponents']).tolist()+['shd',]:
        fid_pars['priors'][cmpt]={};
        for par in fid_pars['free_pars'][cmpt]:
            fid_pars['priors'][cmpt][par] = param_trans[cmpt][par][5:]
            params_i += 1;

    # Gaia selection function applied
    fid_pars['gsf_pars'] = sf_utils.get_gaiasf_pars(theta=fid_pars['lat_min'], nskip=2, _nside=64)

    print(' Free: ', fid_pars['free_pars'][0], ' Free shared: ', fid_pars['free_pars']['shd'], ' Fixed: ', fid_pars['fixed_pars'][0])

    p0 = (np.random.rand(bounds.shape[1]) + bounds[0])/(bounds[1]-bounds[0])

    output['source_id'] = sample['source_id']
    output['true_pars'] = true_pars
    output['param_trans'] = param_trans
    output['free_pars'] = fid_pars['free_pars']
    output['fixed_pars'] = fid_pars['fixed_pars']
    output['chain'] = {}; output['lnprob'] = {}

    if True: # Run model prior initialisation.

        ndim=p0.shape[0]; nwalkers=ndim*4; nstep=200
        def loglike(params):
            return dh_msto.model_prior(params, fid_pars=fid_pars, grad=False, bounds=bounds)
        p0_walkers = np.random.normal(p0, 0.001, size=(nwalkers,ndim))
        prior_sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)
        for pos,lnp,rstate in tqdm.tqdm(prior_sampler.sample(p0_walkers, iterations=nstep), total=nstep):
            pass
        output['chain']['prior'] = prior_sampler.chain
        output['lnprob']['prior'] = prior_sampler.lnprobability
        prior_flatchain = prior_sampler.chain[:,int(nstep/2)::,:].reshape(-1,ndim)


    if True:

        print('Gaia SF fit, no error:')
        sample_2d = np.vstack((1/sample['s'], np.log(1/sample['s']),
                                 sample['sinb'], np.log(np.sqrt(1-sample['sinb']**2)),
                                 sample['m'])).T[sample['gaiasf_subset']].T
        fid_pars['models']=[dh_msto.log_expmodel_grad, dh_msto.log_expmodel_grad, dh_msto.log_halomodel_grad]
        poisson_kwargs_global = {'sample':sample_2d,
                                 'logmodel': dh_msto.logmodel_grad, 'model_integrate':dh_msto.integral_model_gaiaSF_grad,
                                 'param_bounds':bounds, 'gmm':None, 'bins':([0,np.inf],[-np.inf,np.inf]),
                                 'fid_pars':fid_pars, 'model_prior':None}
        print('poisson_like(p0): %.2e' % poisson_like(p0))

        # Run Newton Conjugate-Gradient
        p0 = prior_flatchain[np.random.choice(np.arange(prior_flatchain.shape[0]), size=n_newtoncg)]
        output['chain']['full_noerr_ncg'] = np.zeros((n_newtoncg, ndim))
        output['lnprob']['full_noerr_ncg'] = np.zeros((n_newtoncg))
        for ii in range(n_newtoncg):
            Nfeval=1
            res = minimize(nloglikelihood, p0[ii], method='Newton-CG', jac=True, callback=printx, options={'disp': True})
            output['chain']['full_noerr_ncg'][ii] = res['x']
            output['lnprob']['full_noerr_ncg'][ii] = -res['fun']

        p0 = output['chain']['full_noerr_ncg'][np.argmin(output['lnprob']['full_noerr_ncg'])]
        sampler = samplers.run_mcmc_global(p0, poisson_like, bounds, nstep=nsteps, ncores=ncores, notebook=False, initialise=True)
        output['chain']['full_noerr'] = sampler.chain
        output['lnprob']['full_noerr'] = sampler.lnprobability

    if False:

        print('Full fit, no error:')
        sample_2d = np.vstack((1/sample['s'], np.log(1/sample['s']),
                                 sample['sinb'], np.log(np.sqrt(1-sample['sinb']**2)),
                                 sample['m']))
        fid_pars['models']=[dh_msto.log_expmodel_grad, dh_msto.log_expmodel_grad, dh_msto.log_halomodel_grad]
        poisson_kwargs_global = {'sample':sample_2d,
                                 'logmodel': dh_msto.logmodel_grad, 'model_integrate':dh_msto.integral_model,
                                 'param_bounds':bounds, 'gmm':None, 'bins':([0,np.inf],[-np.inf,np.inf]),
                                 'fid_pars':fid_pars, 'model_prior':None}
        print('poisson_like(p0): %.2e' % poisson_like(p0))

        # Run Newton Conjugate-Gradient
        p0 = prior_flatchain[np.random.choice(np.arange(prior_flatchain.shape[0]), size=n_newtoncg)]
        output['chain']['full_noerr_ncg'] = np.zeros((n_newtoncg, ndim))
        output['lnprob']['full_noerr_ncg'] = np.zeros((n_newtoncg))
        for ii in range(n_newtoncg):
            Nfeval=1
            res = minimize(nloglikelihood, p0[ii], method='Newton-CG', jac=True, callback=printx, options={'disp': True})
            output['chain']['full_noerr_ncg'][ii] = res['x']
            output['lnprob']['full_noerr_ncg'][ii] = -res['fun']

        p0 = output['chain']['full_noerr_ncg'][np.argmin(output['lnprob']['full_noerr_ncg'])]
        sampler = samplers.run_mcmc_global(p0, poisson_like, bounds, nstep=nsteps, ncores=ncores, notebook=False, initialise=True)
        output['chain']['full_noerr'] = sampler.chain
        output['lnprob']['full_noerr'] = sampler.lnprobability


    save_hdf5(output, savefile)
