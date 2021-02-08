import sys, os, pickle, time, warnings

import numpy as np, pandas as pd, scipy, scipy.stats as stats, tqdm, h5py
from copy import deepcopy as copy

sys.path.extend(['../utilities/', '../models/'])
import samplers, disk_cone_plcut as dcp, plotting, transformations
import disk_halo_mstogap as dh_msto
from transformations import func_inv_jac


param_trans = {}
param_trans['shd'] = {'alpha1':('nexp',0,0,-3,3),
                      'alpha2':('nexp',0,0,-3,3)}
param_trans[0] = {'w':('exp',0,0,-10,10),
                  'fD': ('logit_scaled', 0,1, -10,10),
                  'alpha3':('nexp',0,0,-10,10),
                  'hz': ('logit_scaled', 0,  1.2,-10,10)}
param_trans[1] = {'w':('exp',0,0,-10,10),
                  'fD': ('logit_scaled', 0,1,-10,10),
                  'alpha3':('nexp',0,0,-10,10),
                  'hz': ('logit_scaled', 1.2,3,-10,10)}
param_trans[2] = {'w':('exp',0,0,-10,10),
                  'fD': ('logit_scaled', 0,1,-10,10),
                  'alpha3':('nexp',0,0,-10,10),
                  'hz': ('logit_scaled', 3,  7.3,-10,10)}


def poisson_like(params, bounds=None, grad=False):

    poisson_kwargs = copy(poisson_kwargs_global)

    # Prior boundaries
    if bounds is None: bounds = poisson_kwargs['param_bounds']
    if np.sum((params<=bounds[0])|(params>=bounds[1]))>0:
        if grad: return -1e20, np.zeros(len(params))
        else: return -1e20

    # Optional prior inclusion
    if poisson_kwargs_global['model_prior'] is not None:
        prior=poisson_kwargs_global['model_prior'](params, fid_pars=poisson_kwargs['fid_pars'], grad=grad)
    else: prior=0.

    integral = poisson_kwargs['model_integrate'](params, bins=poisson_kwargs['bins'], fid_pars=poisson_kwargs['fid_pars'], grad=grad)
    obj = poisson_kwargs['logmodel'](poisson_kwargs['sample'], params, gmm=poisson_kwargs['gmm'], fid_pars=poisson_kwargs['fid_pars'], grad=grad)
    if not grad: return np.sum(obj) - integral + prior

    elif grad:
        model_val = np.sum(obj[0]) - integral[0] + prior[0]
        model_grad = np.sum(obj[1], axis=1) - integral[1] + prior[1]
        return model_val, model_grad

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

    savefile = "/data/asfe2/Projects/mwtrace_data/mockmodel/fits_allcmpt.h"
    if os.path.exists(savefile): raise ValueError('File: %s already exists!' % savefile)

    nsteps=1000; ncores=2;

    # output dictionary:
    output = {}

    # Load Sample
    size = 1000
    sample = {}; true_pars={}; latent_pars={};
    filename = '/data/asfe2/Projects/mwtrace_data/mockmodel/sample.h'
    with h5py.File(filename, 'r') as hf:
        #subset = (hf['sample']['M'][...]>hf['true_pars'][str(cmpt)]['Mto'][...])
        subsample  = np.sort(np.random.choice(np.arange(len(hf['sample']['source_id'])), size=size, replace=False))
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
    for j in range(3): true_pars[str(cmpt)]['w']*=size

    fid_pars = {'Mmax':true_pars['Mx'],  'lat_min':np.deg2rad(true_pars['theta_deg']), 'R0':true_pars['R0'],
                'free_pars':{}, 'fixed_pars':{}, 'functions':{}, 'functions_inv':{}, 'jacobians':{}, 'w':True,
                'components':['disk','disk','halo'], 'ncomponents':3}

    fid_pars['free_pars'][0] = ['w', 'hz']
    fid_pars['free_pars'][1] = ['w', 'hz']
    fid_pars['free_pars'][2] = ['w', 'hz']
    fid_pars['free_pars']['shd'] = ['alpha1', 'alpha2']

    fid_pars['fixed_pars'][0] = {'Mms':true_pars['Mms'], 'fD':1.-1e-15, 'alpha3':true_pars['0']['alpha3'],
                                 'Mms1':true_pars['Mms1'], 'Mms2':true_pars['Mms2'],
                                 'Mto':true_pars['0'']['Mto']}
    fid_pars['fixed_pars'][1] = copy(fid_pars['fixed_pars'][0]); fid_pars['fixed_pars'][2] = copy(fid_pars['fixed_pars'][0])
    fid_pars['fixed_pars'][1]['Mto'] = true_pars['1']['Mto']
    fid_pars['fixed_pars'][2]['Mto'] = true_pars['2']['Mto']

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

    print(' Free: ', fid_pars['free_pars'][0], ' Free shared: ', fid_pars['free_pars']['shd'], ' Fixed: ', fid_pars['fixed_pars'][0])

    p0 = np.array( [transformations.logit(np.random.rand()),transformations.logit(np.random.rand()),
                    transformations.logit(np.random.rand()),transformations.logit(np.random.rand()),
                    transformations.logit(np.random.rand()),transformations.logit(np.random.rand()),
                    -np.random.rand()*1,-np.random.rand()*1] )

    output['source_id'] = sample['source_id']
    output['true_pars'] = true_pars
    output['param_trans'] = param_trans
    output['free_pars'] = fid_pars['free_pars']
    output['fixed_pars'] = fid_pars['fixed_pars']
    output['chain'] = {}; output['lnprob'] = {}

    if True:
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

        sampler = samplers.run_mcmc_global(p0, poisson_like, bounds, nstep=nsteps, ncores=ncores, notebook=False)
        output['chain']['full_noerr'] = sampler.chain
        output['lnprob']['full_noerr'] = sampler.lnprobability

    #save_hdf5(output, savefile)s

    if True:
        print('Full fil, parallax error:')
        sample_2d = np.vstack((sample['parallax_obs'], sample['parallax_error'],
                               np.abs(sample['sinb']), np.log(np.sqrt(1-sample['sinb']**2)),
                               sample['m'], np.log(sample['parallax_error'])))
        fid_pars['models']=[dh_msto.log_expmodel_perr, dh_msto.log_expmodel_perr, dh_msto.log_halomodel_perr]
        poisson_kwargs_global = {'sample':sample_2d, 'logmodel': dh_msto.logmodel_perr, 'model_integrate':dh_msto.integral_model,
                                 'param_bounds':bounds, 'gmm':None, 'bins':([0,np.inf],[-np.inf,np.inf]),
                                 'fid_pars':fid_pars, 'model_prior':None}
        print('poisson_like(p0): %.2e' % poisson_like(p0))

        sampler = samplers.run_mcmc_global(p0, poisson_like, bounds, nstep=nsteps, ncores=ncores, notebook=False)
        output['chain']['full_perr'] = sampler.chain
        output['lnprob']['full_perr'] = sampler.lnprobability

    save_hdf5(output, savefile)
