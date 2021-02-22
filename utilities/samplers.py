import sys, os
import numpy as np
import emcee, tqdm, corner
from multiprocessing import Pool
from copy import deepcopy as copy

#%% Likelihood functions
def poisson_like(params, sample=None, param_bounds=(-np.inf, np.inf), integration='scipy',
                 logmodel=None, model_integrate=None, bins=None, gmm=None, fid_pars=None, model_prior=None):

    """
    poisson_like: loglikelihood for Poisson probability distribution.
    ----
    args
    ----
    params - model parameters
    logmodel - log of model evaluated at sample coordinates.
    model_integrate - integral of model over coordinates.
    """

    # Prior boundaries
    if np.sum((params<=param_bounds[0])|(params>=param_bounds[1]))>0:
        return -1e20
    # Optional prior inclusion
    if model_prior is not None: prior=model_prior(params)
    else: prior=0.

    integral = model_integrate(params, bins=bins, fid_pars=fid_pars)
    obj = logmodel(sample, params, gmm=gmm, fid_pars=fid_pars)

    return np.sum(obj) - integral + prior


#%% Samplers
def run_mcmc(p0, poisson_kwargs, nstep=1000, ncores=1):

    """
    run_mcmc: emcee parameter exploration on Poisson likelihood function.
    """

    ndim=p0.shape[0]
    nwalkers=ndim*4

    param_bounds=poisson_kwargs['param_bounds']
    p0_walkers = np.random.normal(p0, np.abs(p0/100), size=(nwalkers,ndim))
    unbound = (p0_walkers<=param_bounds[0])|(p0_walkers>=param_bounds[1])
    if np.sum(unbound)>0: print("p0 outside bounds: ", np.sum(unbound))

    with Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, poisson_like, kwargs=poisson_kwargs, pool=pool)
        for pos,lnp,rstate in tqdm.tqdm_notebook(sampler.sample(p0_walkers, iterations=nstep), total=nstep):
            pass

    return sampler

def run_mcmc_global(p0, poisson_like, bounds, nstep=1000, ncores=1, notebook=False, initialise=True):
    """
    run_mcmc_global: emcee with poisson_kwargs defined globally.
    Data is not passed back and forth but is always called as a global argument.
    Leads to large performance increase for big datasets
    """

    if notebook: tqdm_foo = tqdm.tqdm_notebook
    else: tqdm_foo = tqdm.tqdm

    ndim=p0.shape[0]

    # print('bounds', bounds)
    # print('p0', p0)
    # print('p0walker', np.random.normal(p0, np.abs(bounds[1]-bounds[0]).astype(float)/100000))
    if initialise: nwalkers=ndim*4; p0_walkers = np.random.normal(p0, np.abs(bounds[1]-bounds[0]).astype(float)/100000, size=(nwalkers,ndim))
    else: nwalkers=p0.shape[1]; p0_walkers = p0.copy().T

    with Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, poisson_like, pool=pool)
        for pos,lnp,rstate in tqdm_foo(sampler.sample(p0_walkers, iterations=nstep), total=nstep):
            pass

    return sampler




def sample_mcmc1d(sample_like, param_dict, nsample=1000, logmodel=True, show_progress=True):

    ndim=1
    nwalkers=10
    p0 = np.random.rand(1,10).T + 0.1
    if logmodel: sampler = emcee.EnsembleSampler(nwalkers, ndim, sample_like, kwargs=param_dict)
    else: sampler = emcee.EnsembleSampler(nwalkers, ndim, log_samplelike, kwargs={'model':sample_like, 'model_kwargs':param_dict})

    nstep=nsample*10
    if show_progress:
        for _ in tqdm.tqdm_notebook(sampler.sample(p0, iterations=nstep), total=nstep):
            pass
    else:
        sampler.run_mcmc(p0, N=nstep)

    sample = np.random.choice(sampler.chain[:,int(nstep/2):,0].flatten(), size=nsample, replace=False)

    return sample


def sample_mcmcnd(sample_like, param_dict, ndim=1, nsample=1000,
                    logmodel=True, show_progress=True, sranges=None):

    ndim=ndim
    nwalkers=ndim*4
    if sranges is None: p0 = np.random.rand(ndim,nwalkers).T + 0.1
    else: p0 = np.random.rand(ndim, nwalkers).T * (sranges[:,1]-sranges[:,0]) + sranges[:,0]
    if logmodel: sampler = emcee.EnsembleSampler(nwalkers, ndim, sample_like, kwargs=param_dict)
    else: sampler = emcee.EnsembleSampler(nwalkers, ndim, log_samplelike, kwargs={'model':sample_like, 'model_kwargs':param_dict})

    nstep=nsample*10
    if show_progress:
        for pos,lnp,rstate in tqdm.tqdm_notebook(sampler.sample(p0, iterations=nstep), total=nstep):
            pass
    else:
        sampler.run_mcmc(p0, N=nstep)

    chains = sampler.chain[:,int(nstep/2):,:].reshape(-1,ndim)
    sampleid = np.random.choice(np.arange(chains.shape[0]), size=nsample, replace=False)

    return chains[sampleid]

def log_samplelike(x, model=None, model_kwargs={}):
    return np.log(model(x, **model_kwargs))
