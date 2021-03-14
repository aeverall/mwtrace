import sys, os, pickle, time, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, scipy, scipy.stats as stats, tqdm, h5py, emcee
from scipy.optimize import minimize
from copy import deepcopy as copy

sys.path.extend(['../utilities/', '../models/'])
import samplers, disk_cone_plcut as dcp, plotting, transformations, sf_utils
import disk_halo_mstogap as dh_msto
from transformations import func_inv_jac



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


    def gradient_descent(self, p0=None, method='Newton-CG', **model_kwargs):

        def nloglikelihood(x):
            # Negative log likelihood and gradient for Newton-CG
            lnl, grad = poisson_like(x, grad=True)
            return -lnl, -grad

        if p0 is None:
            p0 = self.prior_flatchain[0]

        self._generate_kwargs(p0=p0, **model_kwargs)

        global Nfeval
        Nfeval=1;

        res = scipy.optimize.minimize(nloglikelihood, p0, method='Newton-CG', jac=True, options={'disp': True},
                                     callback=printx)

        return res

    def sample_prior(self):

        p0 = (np.random.rand(self.bounds.shape[1]) + self.bounds[0])/(self.bounds[1]-self.bounds[0])

        ndim=p0.shape[0]; nwalkers=ndim*4; nstep=200
        def loglike(params):
            return dh_msto.model_prior(params, fid_pars=self.fid_pars, grad=False, bounds=self.bounds)

        p0_walkers = np.random.normal(p0, 0.001, size=(nwalkers,ndim))
        prior_sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)
        for pos,lnp,rstate in self.tqdm(prior_sampler.sample(p0_walkers, iterations=nstep), total=nstep):
            pass
        self.prior_flatchain = prior_sampler.chain[:,int(nstep/2)::,:].reshape(-1,ndim)

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
                                 'fid_pars':self.fid_pars, 'model_prior':None}

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

##Print callback function
def printx(Xi):
    global Nfeval
    global fout
    sys.stdout.write('At iterate {0}, {1}'.format(Nfeval, poisson_like(Xi)) + '\r')
    #sys.stdout.write('\r'+str(Xi)+'\n')
    Nfeval += 1

def nloglikelihood(x):
    # Negative log likelihood and gradient for Newton-CG
    lnl, grad = poisson_like(x, grad=True)
    return -lnl, -grad
