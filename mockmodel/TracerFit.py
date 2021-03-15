import sys, os, pickle, time, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, scipy, scipy.stats as stats, tqdm, h5py, emcee
from scipy.optimize import minimize
from copy import deepcopy as copy
from multiprocessing import Pool, Array

sys.path.extend(['../utilities/', '../models/'])
import samplers, disk_cone_plcut as dcp, plotting, transformations, sf_utils
import disk_halo_mstogap as dh_msto
from transformations import func_inv_jac
from functools import partial



global poisson_kwargs_global
poisson_kwargs_global = {}



class mwfit():

    def __init__(self, components=['disk','disk','halo'], free_pars={}, fixed_pars={}, sample={}, sf_bool=False, perr_bool=False):


        self.components=components
        self.free_pars=free_pars
        self.fixed_pars=fixed_pars
        self.sample=sample

        self.sf_bool = sf_bool
        self.perr_bool = perr_bool

        # Function for recording progress
        self.tqdm = tqdm.tqdm

        # Optimizer results
        self.optimize_results = {}
        self.optimize_results['x'] = {}
        self.optimize_results['lnp'] = {}
        # mcmc results
        self.mcmc_results = {}
        self.mcmc_results['chain'] = {}
        self.mcmc_results['lnprob'] = {}

        # Transformations
        # transform, p1, p2, lower bound, upper bound
        self.param_trans = {}
        a_dirichlet = 2
        self.param_trans['shd'] = {'alpha1':('nexp',0,0,-5,3,'none'),
                              'alpha2':('nexp',0,0,-5,3,'none')}
        self.param_trans[0] = {'w':('exp',0,0,-10,10,'dirichlet',a_dirichlet),
                          'fD': ('logit_scaled', 0,1, -10,10,'logistic'),
                          'alpha3':('nexp',0,0,-10,10,'none'),
                          'hz': ('logit_scaled', 0.1,  1.2,-10,10,'logistic')}
        self.param_trans[1] = {'w':('exp',0,0,-10,10,'dirichlet',a_dirichlet),
                          'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                          'alpha3':('nexp',0,0,-10,10,'none'),
                          'hz': ('logit_scaled', 1.2,3,-10,10,'logistic')}
        self.param_trans[2] = {'w':('exp',0,0,-10,10,'dirichlet',a_dirichlet),
                          'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                          'alpha3':('nexp',0,0,-10,10,'none'),
                          'hz': ('logit_scaled', 3,  7.3,-10,10,'logistic')}

        # Output dictionary will be saved
        self.output = {}
        self.output['chain'] = {}; self.output['lnprob'] = {}


    def optimize(self, p0=None, method='Newton-CG', idx_prior=0, label='_', verbose=False, **model_kwargs):

        if p0 is None:
            p0 = self.prior_flatchain[idx_prior]

        self._generate_kwargs(p0=p0, **model_kwargs)

        p0 = self.renormalise(p0)

        global Nfeval
        Nfeval=1;

        res = scipy.optimize.minimize(nloglikelihood, p0, method=method, jac=True, options={'disp': verbose},
                                     callback=printx)

        self.optimize_results['x'][label] = res['x']
        self.optimize_results['lnp'][label] = -res['fun']

        return res

    def optimize_parallel(self, niter=1, p0=None, idxs_prior=None, method='Newton-CG', ncores=1, label='_', verbose=False, **model_kwargs):

        if p0 is None:
            if idxs_prior is None:
                idxs_prior=np.random.choice(np.arange(self.prior_flatchain.shape[0]), niter, replace=False)
            p0 = self.prior_flatchain[idxs_prior]

        self._generate_kwargs(p0=p0[0], **model_kwargs)

        p0 = np.array([self.renormalise(p0[i]) for i in range(niter)])
        p0 = list(zip(np.arange(niter), p0))

        global Nfeval
        Nfeval = Array('i', np.zeros(niter, dtype=int))
        global fout
        fout = Array('f', np.zeros(niter, dtype=float))

        def run_gradient_descent(i):
            res = scipy.optimize.minimize(nloglikelihood, p0[i])
            return i, res['x']

        #res = maximize(p0[0], p0_idx=True, method=method, jac=True, options={'disp': True}, callback=printx_set)
        result = np.zeros((niter, len(p0[0][1]))); lnprob = np.zeros(niter); i=0
        kwargs = {}
        with Pool(ncores) as pool:
            for res in pool.imap(partial(maximize, p0_idx=True, method=method, jac=True, options={'disp': verbose}, callback=printx_set), p0):
                print(i)
                result[i] = res['x']
                lnprob[i] = -res['fun']
                i+=1

        self.optimize_results['x'][label] = result
        self.optimize_results['lnp'][label] = lnprob

        return result

    def mcmc(self, p0=None, ncores=1, nsteps=1000, label='_', optimize_label='_', **model_kwargs):

        if p0 is None:
            p0 = self.optimize_results['x'][optimize_label][np.argmax(self.optimize_results['lnp'][optimize_label])]

        self._generate_kwargs(p0=p0, **model_kwargs)

        sampler = samplers.run_mcmc_global(p0, poisson_like, self.poisson_kwargs['param_bounds'], nstep=nsteps, ncores=ncores, tqdm_foo=self.tqdm, initialise=True)

        self.mcmc_results['chain'][label] = sampler.chain
        self.mcmc_results['lnprob'][label] = sampler.lnprobability

    def mcmc_prior(self):

        p0 = (np.random.rand(self.bounds.shape[1]) + self.bounds[0])/(self.bounds[1]-self.bounds[0])

        ndim=p0.shape[0]; nwalkers=ndim*4; nstep=200
        def loglike(params):
            return dh_msto.model_prior(params, fid_pars=self.fid_pars, grad=False, bounds=self.bounds)

        p0_walkers = np.random.normal(p0, 0.001, size=(nwalkers,ndim))
        prior_sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)
        for pos,lnp,rstate in self.tqdm(prior_sampler.sample(p0_walkers, iterations=nstep), total=nstep):
            pass
        self.prior_flatchain = prior_sampler.chain[:,int(nstep/2)::,:].reshape(-1,ndim)

    def renormalise(self, p):

        params = p.copy()

        integral = self.poisson_kwargs['model_integrate'](params, bins=self.poisson_kwargs['bins'], fid_pars=self.poisson_kwargs['fid_pars'], grad=False)
        sample_size = self.poisson_kwargs['sample'].shape[1]

        renorm = sample_size/integral

        free_pars = self.fid_pars['free_pars']
        params_i = 0
        for j in range(self.fid_pars['ncomponents']):
            for par in free_pars[j]:
                if par=='w': params[params_i] = self.fid_pars['functions_inv'][j][par]( self.fid_pars['functions'][j][par](params[params_i]) * renorm )
                params_i += 1;
        for par in free_pars['shd']: params_i += 1

        return params

    def _generate_fid_pars(self):

        fid_pars = {'Mmax':self.fixed_pars['Mx'],  'lat_min':np.deg2rad(self.fixed_pars['theta_deg']), 'R0':self.fixed_pars['R0'],
                    'free_pars':{}, 'fixed_pars':{}, 'functions':{}, 'functions_inv':{}, 'jacobians':{}, 'w':True,
                    'components':self.components, 'ncomponents':len(self.components)}

        fid_pars['free_pars'] = self.free_pars

        fid_pars['fixed_pars'][0] = {'Mms':self.fixed_pars['Mms'], 'fD':self.fixed_pars['0']['fD'], 'alpha3':self.fixed_pars['0']['alpha3'],
                                     'Mms1':self.fixed_pars['Mms1'], 'Mms2':self.fixed_pars['Mms2'],
                                     'Mto':self.fixed_pars['0']['Mto']}
        fid_pars['fixed_pars'][1] = copy(fid_pars['fixed_pars'][0]); fid_pars['fixed_pars'][2] = copy(fid_pars['fixed_pars'][0])
        fid_pars['fixed_pars'][1]['Mto'] = self.fixed_pars['1']['Mto']
        fid_pars['fixed_pars'][1]['fD'] = self.fixed_pars['1']['fD']
        fid_pars['fixed_pars'][2]['Mto'] = self.fixed_pars['2']['Mto']
        fid_pars['fixed_pars'][2]['fD'] = self.fixed_pars['2']['fD']

        fid_pars['functions']={}; fid_pars['functions_inv']={}; fid_pars['jacobians']={}; bounds=[]
        params_i = 0
        for cmpt in np.arange(fid_pars['ncomponents']).tolist()+['shd',]:
            fid_pars['functions'][cmpt]={}; fid_pars['functions_inv'][cmpt]={}; fid_pars['jacobians'][cmpt]={}
            for par in fid_pars['free_pars'][cmpt]:
                fid_pars['functions'][cmpt][par], \
                fid_pars['functions_inv'][cmpt][par], \
                fid_pars['jacobians'][cmpt][par]=func_inv_jac[self.param_trans[cmpt][par][0]](*self.param_trans[cmpt][par][1:3])
                bounds.append([self.param_trans[cmpt][par][3], self.param_trans[cmpt][par][4]])
                params_i += 1;
        self.bounds = np.array(bounds).T

        fid_pars['priors'] = {}
        params_i = 0
        for cmpt in np.arange(fid_pars['ncomponents']).tolist()+['shd',]:
            fid_pars['priors'][cmpt]={};
            for par in fid_pars['free_pars'][cmpt]:
                fid_pars['priors'][cmpt][par] = self.param_trans[cmpt][par][5:]
                params_i += 1;

        # Gaia selection function applied
        if self.sf_bool: fid_pars['gsf_pars'] = sf_utils.get_gaiasf_pars(theta=fid_pars['lat_min'], nskip=2, _nside=64)

        self.fid_pars=fid_pars

    def _generate_kwargs(self, p0=None):

        print('SF - %s, Parallax error - %s' % (str(self.sf_bool), str(self.perr_bool)))

        self.poisson_kwargs = {'param_bounds':self.bounds, 'gmm':None, 'bins':([0,np.inf],[-np.inf,np.inf]),
                                 'fid_pars':self.fid_pars, 'model_prior':dh_msto.model_prior}

        if not self.perr_bool:
            sample_2d = np.vstack((1/self.sample['s'], np.log(1/self.sample['s']),
                                     self.sample['sinb'], np.log(np.sqrt(1-self.sample['sinb']**2)),
                                     self.sample['m']))
            self.fid_pars['models']=[dh_msto.log_expmodel_grad, dh_msto.log_expmodel_grad, dh_msto.log_halomodel_grad]
            self.poisson_kwargs['logmodel'] = dh_msto.logmodel_grad
        else:
            sample_2d = np.vstack((self.sample['parallax_obs'], self.sample['parallax_error'],
                                   np.abs(self.sample['sinb']), np.log(np.sqrt(1-self.sample['sinb']**2)),
                                   self.sample['m'], np.log(self.sample['parallax_error'])))
            self.fid_pars['models']=[dh_msto.log_expmodel_perr_grad, dh_msto.log_expmodel_perr_grad, dh_msto.log_halomodel_perr_grad]
            self.poisson_kwargs['logmodel'] = dh_msto.logmodel_perr_grad

        if not self.sf_bool:
            self.poisson_kwargs['sample'] = sample_2d
            self.poisson_kwargs['model_integrate'] = dh_msto.integral_model
        elif self.sf_bool:
            self.poisson_kwargs['sample'] = sample_2d.T[self.sample['gaiasf_subset']].T
            self.poisson_kwargs['model_integrate'] = dh_msto.integral_model_gaiaSF_grad

        global poisson_kwargs_global
        poisson_kwargs_global = copy(self.poisson_kwargs)

        if p0 is not None: print('poisson_like(p0): %.2e' % poisson_like(p0))

    def evaluate_likelihood(self, x, **kwargs):

        return poisson_like(x, **kwargs)

    def test_gradient(self, x, delta=1e-8, **kwargs):

        grad = lambda x: poisson_like(x, grad=True, **kwargs)[1]
        model = lambda x: poisson_like(x, grad=True, **kwargs)[0]

        return scipy.optimize.approx_fprime(x, model, delta), grad(x)

    def save_hdf5_recurrent(self, obj, file, path, hf):

        for key, item in obj.items():
            if isinstance(item, dict): self.save_hdf5_recurrent(item, file, os.path.join(path, str(key)), hf)
            else:
                try: hf.create_dataset(os.path.join(path, str(key)), data=item)
                except TypeError: hf.create_dataset(os.path.join(path, str(key)), data=np.array(item).astype('S20'))

    def save_hdf5(self, chain_dict, filename, mode='w'):
        # Save all chains
        print('Saving...' + filename)
        if os.path.exists(filename) and mode=='w':
            raise ValueError('File %s already exists...')
        print('Mode: %s' % mode)

        with h5py.File(filename, mode) as hf:
            self.save_hdf5_recurrent(chain_dict, filename, "", hf)

    def save(self, filename, true_pars, mode='w'):

        # Dictionary to be saved

        # Identifiers of used sources
        self.output['source_id'] = self.sample['source_id']
        # Dictionary of parameters
        if true_pars is not None: self.output['true_pars'] = true_pars
        self.output['param_trans'] = self.param_trans
        self.output['free_pars'] = self.fid_pars['free_pars']
        self.output['fixed_pars'] = self.fid_pars['fixed_pars']
        # Optimization results
        self.output['optimize'] = self.optimize_results
        self.output['mcmc'] = self.mcmc_results

        self.save_hdf5(self.output, filename, mode=mode)

    def get_true_params(self, true_pars):

        """ true_pars dict -> true_params array of free parameters (untransformed). """

        true_params=[];
        for j in range(self.fid_pars['ncomponents']):
            for par in self.fid_pars['free_pars'][j]:
                true_params += [true_pars[str(j)][par],]
        for par in self.fid_pars['free_pars']['shd']:
            true_params += [true_pars[par],]
        true_params=np.array(true_params)

        return true_params

    def transform_params(self, params, transform='functions_inv'):

        """
        params 1D array -> params_f 1D array - Applies parameter transformation to params example.
        Keywords:
        transform="functions_inv" - if "functions_inv", transforms real parameters to likelihood model parameters.
                                  - if "functions", transforms likelihood model parameters to real parameters."""

        params_i = 0
        params_f = []
        for j in range(self.fid_pars['ncomponents']):
            for par in self.fid_pars['free_pars'][j]:
                params_f   += [self.fid_pars[transform][j][par](params[params_i]),]
                params_i += 1
        for par in self.fid_pars['free_pars']['shd']:
            params_f += [self.fid_pars[transform]['shd'][par](params[params_i]),]
            params_i += 1
        params_f=np.array(params_f)

        return params_f


def nloglikelihood(x, id=-1):
    # Negative log likelihood and gradient for Newton-CG
    lnl, grad = poisson_like(x, grad=True)
    if id!=-1:
        global opt_id
        opt_id=id
    return -lnl, -grad

def maximize(p0, p0_idx=False, **kwargs):
    if p0_idx: id=p0[0]; p0=p0[1];
    else: id=-1
    res = scipy.optimize.minimize(nloglikelihood, p0, **kwargs, args=(id))
    return res


def poisson_like(params, bounds=None, grad=False):

    poisson_kwargs = copy(poisson_kwargs_global)

    # Prior boundaries
    if bounds is None: bounds = poisson_kwargs['param_bounds']

    # Optional prior inclusion
    if poisson_kwargs_global['model_prior'] is not None:
        prior=poisson_kwargs_global['model_prior'](params, fid_pars=poisson_kwargs['fid_pars'], grad=grad, bounds=bounds)
    else:
        if np.sum((params<=bounds[0])|(params>=bounds[1]))>0:
            if grad: return -1e20, np.zeros(len(params))
            else: return -1e20

    integral = poisson_kwargs['model_integrate'](params, bins=poisson_kwargs['bins'], fid_pars=poisson_kwargs['fid_pars'], grad=grad)
    obj = poisson_kwargs['logmodel'](poisson_kwargs['sample'], params, gmm=poisson_kwargs['gmm'], fid_pars=poisson_kwargs['fid_pars'], grad=grad)
    if not grad: return np.sum(obj) - integral + prior

    elif grad:
        model_val = np.sum(obj[0]) - integral[0] + prior[0]
        model_grad = np.sum(obj[1], axis=1) - integral[1] + prior[1]
        return model_val, model_grad

##Print callback function
def printx(Xi):
    global Nfeval
    sys.stdout.write('At iterate {0}, {1}'.format(Nfeval, poisson_like(Xi)) + '\r')
    Nfeval += 1
def printx_set(Xi):
    global opt_id
    fout[opt_id] = poisson_like(Xi)
    Nfeval[opt_id] += 1
    sys.stdout.write(', '.join(['{0}:{1:.5e}'.format(np.array(Nfeval)[i],np.array(fout)[i]) for i in range(len(np.array(Nfeval)))])+'\r')
    sys.stdout.flush()