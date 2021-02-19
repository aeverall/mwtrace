import sys, os
import numpy as np, h5py, scipy, scipy.stats

sys.path.append('../utilities/')
import distributions
import samplers



def Mbol_func(m, a=4):
    return -2.5*a*np.log(m)/np.log(10)

if __name__=='__main__':

    # Input parameters
    true_pars = {'N':50000,
                 'hz':0.5, 'lat_min':np.pi/3,
                 'ep':2, 'a':4, 'm_min':0.1}
    # Derivatives of input parameters
    latent_pars = {'kappa':(true_pars['ep']-1)*np.log(10)/(2.5*true_pars['a']),
                   'Mbol_max':Mbol_func(true_pars['m_min'], a=true_pars['a'])}


    # Generate population
    _sample = {}

    # Distance - latitude distribution
    sranges = np.array([[0,100],
                        [true_pars['lat_min'], np.pi/2]])
    edsd_params = {'hz':true_pars['hz'], 'N':true_pars['N'], 'theta':true_pars['lat_min'], 'sminmax':(0, np.inf), 'bminmax':(-np.pi/2, np.pi/2)}
    _sample['s'], _sample['b'] = samplers.sample_mcmcnd(distributions.lnvcone, param_dict=edsd_params, ndim=2,
                                             nsample=true_pars['N'], sranges=sranges).T

    # Absolute magnitude distribution
    Mbol_params = {'gamma':true_pars['ep'], 'xmin':true_pars['m_min'], 'a':true_pars['a']}
    _sample['Mbol'] = samplers.sample_mcmc1d(distributions.Mbol_dist, logmodel=False,
                                             param_dict=Mbol_params, nsample=true_pars['N'])

    # Apparent Magnitude
    _sample['m'] = _sample['Mbol'] + 5*np.log10(100*_sample['s'])


    # Observables - draws from error distribution
    # Observed parallax
    _sample['parallax_error'] = scipy.stats.gamma.rvs(1., size=true_pars['N'])/2.
    _sample['parallax'] = 1/_sample['s']
    _sample['parallax_obs'] = np.random.normal(_sample['parallax'], _sample['parallax_error'])

    # Observed apparent magnitude
    _sample['m_err'] = scipy.stats.gamma.rvs(1.5, size=true_pars['N'])/3.
    _sample['m_obs'] = np.random.normal(_sample['m'], _sample['m_err'])


    # Save toy model dataset
    filename = '/data/asfe2/Projects/mwtrace_data/toymodel/sample.h'
    with h5py.File(filename, 'w') as hf:
        for key in true_pars.keys():
            hf.create_dataset('true_pars/'+key, data=true_pars[key])
        for key in latent_pars.keys():
            hf.create_dataset('latent_pars/'+key, data=latent_pars[key])
        for key in _sample.keys():
            hf.create_dataset('sample/'+key, data=_sample[key])
