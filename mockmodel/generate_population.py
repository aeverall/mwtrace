"""
Generate sample from three component disk model:
-Exponential disk
-Exponential disk
-Power law halo
"""

import sys, os, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np, h5py, scipy, scipy.stats, emcee, tqdm
from multiprocessing import Pool
from copy import deepcopy as copy


def dwarf_magnitude(M, alpha1=-0.5, alpha2=-1.,
                 Mto=4., Mms=8., Mms1=9., Mms2=7., Mx=10.):

    M = M[0]
    if (M<Mto)|(M>Mx): return -1e30

    ep1=1.3; ep2=2.3;
    a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    if M>Mms1: return - np.log(a1) + alpha1*(Mms-M)
    elif M>Mms2: return np.log(Ag) + alphag*(Mms-M)
    elif M>Mto: return - np.log(a2) + alpha2*(Mms-M)
    else: raise ValueError('No good mag range')

def giant_magnitude(M, alpha3=-0.5, Mto=4.):

    M = M[0]
    if (M>Mto): return -1e30

    else: return -alpha3*M

def disk_pos(s, sinb, hz=1., R0=8.27, theta_deg=60):

    if (sinb<np.sin(np.deg2rad(theta_deg)))|(sinb>1)|(s<0): return -1e30

    return 2*np.log(s) - s*sinb/hz

def halo_pos(s, sinb, hz=1., R0=8.27, theta_deg=60):

    if (sinb<np.sin(np.deg2rad(theta_deg)))|(sinb>1)|(s<0): return -1e30

    return 2*np.log(s) - hz/2*np.log((s*sinb)**2 + R0**2)

if __name__=='__main__':

    #%% Initialis model parameters

    burnt_samples = {}
    for j in range(3):
        burnt_samples[j]={}

    selected_samples = {}
    for j in range(3):
        selected_samples[j]={}

    nsample = 100000
    ncores=2
    nstep=int(nsample/30 * 2 * 5)
    print('Nsample: ', nsample)


    alpha1=-0.15; alpha2=-0.3
    Mms=8.; Mms1=9.; Mms2=7.; Mx=10.7
    R0=8.27; theta_deg=60
    # Parameters for all three components of the model.
    global_params = {'alpha1':-0.15, 'alpha2':-0.3,
                    'Mms':8., 'Mms1':9., 'Mms2':7., 'Mx':10.7,
                    'R0':8.27, 'theta_deg':60, 'N':nsample}
    # Individual independent component parameters.
    params = {0: {'hz':0.9, 'alpha3':-1.,  'Mto':4.8, 'fD':0.94, 'w':0.2},
              1: {'hz':1.9, 'alpha3':-0.5, 'Mto':3.14, 'fD':0.998, 'w':0.3},
              2: {'hz':4.6, 'alpha3':-0.6,  'Mto':3.3, 'fD':0.995,  'w':0.5}}
    print(params)

    weights = np.array([params[j]['w'] for j in range(3)])
    j_nsample = np.round(weights * nsample).astype(int)
    j_nsample[-1] = nsample-np.sum(j_nsample[:-1])

    fdwarf = np.array([params[j]['fD'] for j in range(3)])
    j_ndwarf = np.round(j_nsample * fdwarf).astype(int)
    j_ngiant = j_nsample - np.round(j_nsample * fdwarf).astype(int)


    nwalkers=30; ndim=1;
    for j in range(3):

        print('Dwarf %d' % j)
        def loglike(x):
            return dwarf_magnitude(x, alpha1=global_params['alpha1'], alpha2=global_params['alpha2'],
                                 Mto=params[j]['Mto'], Mms=global_params['Mms'],
                                 Mms1=global_params['Mms1'], Mms2=global_params['Mms2'], Mx=global_params['Mx'])

        p0_walkers = np.random.rand(nwalkers,1) * (global_params['Mms1']-params[j]['Mto']) + params[j]['Mto']

        with Pool(ncores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, pool=pool)
            for pos,lnp,rstate in tqdm.tqdm(sampler.sample(p0_walkers, iterations=nstep), total=nstep):
                pass

        burnt_samples[j]['M_dwarfs'] = np.reshape(sampler.chain[:, int(nstep/2):, :][:,::3,:], (-1,1)).copy().T[0]

    for j in range(3):
        print('Giant %d' % j)
        def loglike(x):
            return giant_magnitude(x, alpha3=params[j]['alpha3'], Mto=params[j]['Mto'])

        p0_walkers = np.random.rand(nwalkers,1) * params[j]['Mto']

        with Pool(ncores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, pool=pool)
            for pos,lnp,rstate in tqdm.tqdm(sampler.sample(p0_walkers, iterations=nstep), total=nstep):
                pass

        burnt_samples[j]['M_giants'] = np.reshape(sampler.chain[:, int(nstep/2):, :][:,::3,:], (-1,1)).copy().T[0]


    for j in range(3):
        # ndwarf = int(nsample * params[j]['w'] * params[j]['fD'])
        # ngiant = int(nsample * params[j]['w'] * (1-params[j]['fD']))

        dwarf_sample = np.random.choice(np.arange(burnt_samples[j]['M_dwarfs'].shape[0]), j_ndwarf[j], replace=False)
        giant_sample = np.random.choice(np.arange(burnt_samples[j]['M_giants'].shape[0]), j_ngiant[j], replace=False)

        order = np.random.choice(np.arange(len(dwarf_sample)+len(giant_sample)),
                                      size=len(dwarf_sample)+len(giant_sample), replace=False)
        selected_samples[j]['M'] = np.hstack((burnt_samples[j]['M_dwarfs'][dwarf_sample],
                                              burnt_samples[j]['M_giants'][giant_sample]))[order]
        selected_samples[j]['Dwarf'] = np.hstack((np.ones(len(dwarf_sample)).astype(int),
                                                  np.zeros(len(giant_sample)).astype(int)))[order]


    #%% Position distribution sampling
    functions = [disk_pos, disk_pos, halo_pos]

    nwalkers=30; ndim=2;
    for j in range(3):
        print('Spatial %d' % j)
        def loglike(x):
            return functions[j](x[0], x[1], hz=params[j]['hz'], R0=global_params['R0'], theta_deg=global_params['theta_deg'])

        p0_walkers = np.random.rand(nwalkers,ndim)
        p0_walkers *= 1-np.sin(np.deg2rad(global_params['theta_deg']))
        p0_walkers += np.sin(np.deg2rad(global_params['theta_deg']))

        with Pool(ncores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, pool=pool)
            for pos,lnp,rstate in tqdm.tqdm(sampler.sample(p0_walkers, iterations=nstep), total=nstep):
                pass

        burnt_samples[j]['x'] = np.reshape(sampler.chain[:, int(nstep/2):, :][:,::3,:], (-1,ndim)).copy().T

    for j in range(3):
        cmpt_sample = np.random.choice(np.arange(burnt_samples[j]['x'].shape[1]), j_nsample[j], replace=False)

        selected_samples[j]['s'] = burnt_samples[j]['x'][:,cmpt_sample][0]
        selected_samples[j]['sinb'] = burnt_samples[j]['x'][:,cmpt_sample][1]
        selected_samples[j]['l'] = np.random.rand(len(cmpt_sample))*2*np.pi

    for key in ['s', 'sinb', 'l', 'M']:
        selected_samples[key] = np.hstack([selected_samples[j][key] for j in range(3)])
    selected_samples['cmpt'] = np.hstack([np.zeros(len(selected_samples[j][key])).astype(int) + j for j in range(3)])


    # Apparent Magnitude
    selected_samples['m'] = selected_samples['M'] + 5*np.log10(100*selected_samples['s'])

    # Observables - draws from error distribution
    # Observed parallax
    selected_samples['parallax_error'] = scipy.stats.gamma.rvs(1., size=global_params['N'])/2.
    selected_samples['parallax'] = 1/selected_samples['s']
    selected_samples['parallax_obs'] = np.random.normal(selected_samples['parallax'], selected_samples['parallax_error'])

    # Observed apparent magnitude
    selected_samples['m_err'] = scipy.stats.gamma.rvs(1.5, size=global_params['N'])/3.
    selected_samples['m_obs'] = np.random.normal(selected_samples['m'], selected_samples['m_err'])


    #%% Save data in HDF5 format
    filename = '/data/asfe2/Projects/mwtrace_data/mockmodel/sample.h'
    print('Saving...' + filename)
    with h5py.File(filename, 'w') as hf:
            for k in global_params.keys():
                hf.create_dataset('true_pars/'+k, data=global_params[k])
            for j in range(3):
                for k in params[j].keys():
                    hf.create_dataset('true_pars/'+str(j)+'/'+k, data=params[j][k])
            for k in ['s', 'sinb', 'l', 'M', 'cmpt',
                      'm','parallax_error', 'parallax_obs','m_err','m_obs']:
                print(k, len(selected_samples[k]))
                hf.create_dataset('sample/'+k, data=selected_samples[k])
            hf.create_dataset('sample/source_id', data=np.arange(len(selected_samples['s']))
