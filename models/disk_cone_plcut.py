"""
disk_cone_plcut: Tracer Density model module.

- Exponential disk profile.
- Observing high latitudes with a projected cone.
- Power-law absolute magnitude distribution:
    - Positive power.
    - High (dim) cut-off.
"""

import sys
sys.path.append('../utilities/')
import functions
import numpy as np, scipy


def integrand(p, beta, pi_mu, pi_err, n, b=None):
    return p**n * np.exp(-beta/p - ((p-pi_mu)**2/(2*pi_err**2)))

def logmodel(sample, params, gmm=None, fid_pars=None):

    # Observables
    pi_mu, lat, m_mu = sample

    # Parameters
    logN = params[0]
    hz = 1/np.exp(params[1])
    alpha = params[2]
    Mmax = fid_pars['Mbol_max_true']
    theta= fid_pars['lat_min']

    # Latent variables
    n = -4 + alpha
    k = alpha*np.log(10)/5
    beta = np.exp(params[1])*np.abs(np.sin(lat))

    # Normalisation of distributions
    log_mnorm = np.log(k) - k*(Mmax+10)
    log_pnorm = 2*np.log(np.tan(theta)) - 3*np.log(hz)

    # Distributions
    log_m = k*m_mu
    log_p = n*np.log(pi_mu) + np.log(np.cos(lat)) - np.abs(np.sin(lat))/(hz*pi_mu)

    return logN + log_mnorm + log_m + log_pnorm + log_p

def logmodel_perr(sample, params, gmm=None, fid_pars=None):

    # Observables
    pi_mu, lat, m_mu, pi_err = sample

    # Parameters
    logN = params[0]
    hz = 1/np.exp(params[1])
    alpha = params[2]
    Mmax = fid_pars['Mbol_max_true']
    theta= fid_pars['lat_min']

    # Latent variables
    n = -4 + alpha
    k = alpha*np.log(10)/5
    beta = np.exp(params[1])*np.abs(np.sin(lat))
    # Max parallax due to minimum abs mag
    pi_max = 10**((Mmax+10-m_mu)/5)

    # Normalisation of distributions
    log_mnorm = np.log(k) - k*(Mmax+10)
    log_pnorm = 2*np.log(np.tan(theta)) - 3*np.log(hz) - 0.5*np.log(2*np.pi*pi_err**2)

    # Magnitude contribution
    log_m = k*m_mu
    # Parallax contribution
    kwargs = {'transform':'logit', 'b':pi_max}
    coeffs = np.array([np.zeros(len(pi_mu))+1., - (pi_max+pi_mu), (pi_max*pi_mu-(n+2)*pi_err**2),
                       (pi_max*(n+1)-beta)*pi_err**2, pi_max*beta*pi_err**2]).T + 0.j
    root, ngood = functions.get_roots(coeffs, pi_max)
    root_z = functions.trans(root, **kwargs)
    if np.sum(ngood>1)>0:
        print(np.unique(ngood), np.sum(ngood>1))
    args = (beta, pi_mu, pi_err, n)
    curve = functions.d2logIJ_dp2(root, *args, **kwargs)/functions.jac(root, **kwargs)**2
    log_p = np.log(functions.integrate_gh(integrand,
                                              functions.jac,
                                              functions.trans_i,
                                              root_z, 1/np.sqrt(-curve), args, kwargs))

    return logN + log_mnorm + log_pnorm + log_m + log_p

def logmodel_perr_ridder(sample, params, gmm=None, fid_pars=None, get_coeffs=False):

    # Observables
    pi_mu, lat, m_mu, pi_err = sample

    # Parameters
    logN = params[0]
    hz = 1/np.exp(params[1])
    alpha = params[2]
    Mmax = fid_pars['Mbol_max_true']
    theta= fid_pars['lat_min']

    # Latent variables
    n = -4 + alpha
    k = alpha*np.log(10)/5
    beta = np.exp(params[1])*np.abs(np.sin(lat))
    # Max parallax due to minimum abs mag
    pi_max = 10**((Mmax+10-m_mu)/5)

    # Normalisation of distributions
    log_mnorm = np.log(k) - k*(Mmax+10)
    log_pnorm = 2*np.log(np.tan(theta)) - 3*np.log(hz) - 0.5*np.log(2*np.pi*pi_err**2)

    # Magnitude contribution
    log_m = k*m_mu
    # Parallax contribution
    kwargs = {'transform':'logit', 'b':pi_max}
    coeffs = np.array([np.zeros(len(pi_mu))+1., - (pi_max+pi_mu), (pi_max*pi_mu-(n+2)*pi_err**2),
                       (pi_max*(n+1)-beta)*pi_err**2, pi_max*beta*pi_err**2]).T
    root = functions.get_roots_ridder_hm(coeffs, b=pi_max)
    root_z = functions.trans(root, **kwargs)
    args = (beta, pi_mu, pi_err, n)
    if get_coeffs: return root, coeffs, kwargs, args
    curve = functions.d2logIJ_dp2(root, *args, **kwargs)/functions.jac(root, **kwargs)**2
    log_p = np.log(functions.integrate_gh(integrand,
                                              functions.jac,
                                              functions.trans_i,
                                              root_z, 1/np.sqrt(-curve), args, kwargs))
    #print(curve, root, root_z)

    return logN + log_mnorm + log_pnorm + log_m + log_p

def logmodel_perr_merr(sample, params, gmm=None, fid_pars=None):

    # Observables
    pi_mu, lat, m_mu, pi_err, m_err = sample

    # Parameters
    logN = params[0]
    hz = 1/np.exp(params[1])
    alpha = params[2]
    Mmax = fid_pars['Mbol_max_true']
    theta= fid_pars['lat_min']

    # Latent variables
    n = -4 + alpha
    k = alpha*np.log(10)/5
    beta = np.exp(params[1])*np.abs(np.sin(lat))
    # Max parallax due to minimum abs mag
    pi_max = 10**((Mmax+10-(m_mu+k*m_err**2))/5)

    # Normalisation of distributions
    log_mnorm = np.log(k) - k*(Mmax+10)
    log_pnorm = 2*np.log(np.tan(theta)) - 3*np.log(hz) - 0.5*np.log(2*np.pi*pi_err**2)

    # Magnitude contribution
    log_m = k*m_mu + (k**2*m_err**2)/2
    # Parallax contribution
    kwargs = {'transform':'logit', 'b':pi_max}
    coeffs = np.array([np.zeros(len(pi_mu))+1., - (pi_max+pi_mu), (pi_max*pi_mu-(n+2)*pi_err**2),
                       (pi_max*(n+1)-beta)*pi_err**2, pi_max*beta*pi_err**2]).T + 0.j
    root, ngood = functions.get_roots(coeffs, pi_max)
    root_z = functions.trans(root, **kwargs)
    if np.sum(ngood>1)>0:
        print(np.unique(ngood), np.sum(ngood>1))
    args = (beta, pi_mu, pi_err, n)
    curve = functions.d2logIJ_dp2(root, *args, **kwargs)/functions.jac(root, **kwargs)**2
    log_p = np.log(functions.integrate_gh(integrand,
                                              functions.jac,
                                              functions.trans_i,
                                              root_z, 1/np.sqrt(-curve), args, kwargs))

    return logN + log_mnorm + log_pnorm + log_m + log_p

def logmodel_perr_merr_ridder(sample, params, gmm=None, fid_pars=None, get_coeffs=False):

    # Observables
    pi_mu, lat, m_mu, pi_err, m_err = sample

    # Parameters
    logN = params[0]
    hz = 1/np.exp(params[1])
    alpha = params[2]
    Mmax = fid_pars['Mbol_max_true']
    theta= fid_pars['lat_min']

    # Latent variables
    n = -4 + alpha
    k = alpha*np.log(10)/5
    beta = np.exp(params[1])*np.abs(np.sin(lat))
    # Max parallax due to minimum abs mag
    pi_max = 10**((Mmax+10-(m_mu+k*m_err**2))/5)

    # Normalisation of distributions
    log_mnorm = np.log(k) - k*(Mmax+10)
    log_pnorm = 2*np.log(np.tan(theta)) - 3*np.log(hz) - 0.5*np.log(2*np.pi*pi_err**2)

    # Magnitude contribution
    log_m = k*m_mu + (k**2*m_err**2)/2
    # Parallax contribution
    kwargs = {'transform':'logit', 'b':pi_max}
    coeffs = np.array([np.zeros(len(pi_mu))+1., - (pi_max+pi_mu), (pi_max*pi_mu-(n+2)*pi_err**2),
                       (pi_max*(n+1)-beta)*pi_err**2, pi_max*beta*pi_err**2]).T
    root = functions.get_roots_ridder_hm(coeffs, b=pi_max)
    root_z = functions.trans(root, **kwargs)
    if get_coeffs: return root, coeffs, kwargs, args
    args = (beta, pi_mu, pi_err, n)
    curve = functions.d2logIJ_dp2(root, *args, **kwargs)/functions.jac(root, **kwargs)**2
    log_p = np.log(functions.integrate_gh(integrand,
                                              functions.jac,
                                              functions.trans_i,
                                              root_z, 1/np.sqrt(-curve), args, kwargs))

    return logN + log_mnorm + log_pnorm + log_m + log_p

def logmodel_perr_merr_ridder_GH(sample, params, gmm=None, fid_pars=None):

    # Observables
    pi_mu, lat, m_mu, pi_err, m_err = sample

    # Parameters
    logN = params[0]
    hz = 1/np.exp(params[1])
    alpha = params[2]
    Mmax = fid_pars['Mbol_max_true']
    theta= fid_pars['lat_min']

    # Latent variables
    n = -4 + alpha
    k = alpha*np.log(10)/5
    beta = np.exp(params[1])*np.abs(np.sin(lat))

    # Normalisation of distributions
    log_mnorm = np.log(k) - k*(Mmax+10)
    log_pnorm = 2*np.log(np.tan(theta)) - 3*np.log(hz) - 0.5*np.log(2*np.pi*pi_err**2)

    # Magnitude contribution
    log_m = k*m_mu + (k**2*m_err**2)/2

    # GH approximation nodes and weights
    degree=5
    nodes, weights = np.polynomial.hermite.hermgauss(degree)
    m_nodes = (nodes[:,np.newaxis]*m_err[np.newaxis,:])+m_mu[np.newaxis,:]
    # Array for integral
    I=np.zeros((degree, sample.shape[1]))

    for i in range(degree):
        # Max parallax due to minimum abs mag
        pi_max = 10**((Mmax+10-(m_nodes[i]+k*m_err**2))/5)

        # Parallax contribution
        kwargs = {'transform':'logit', 'b':pi_max}
        coeffs = np.array([np.zeros(len(pi_mu))+1., - (pi_max+pi_mu), (pi_max*pi_mu-(n+2)*pi_err**2),
                           (pi_max*(n+1)-beta)*pi_err**2, pi_max*beta*pi_err**2]).T
        root = functions.get_roots_ridder_hm(coeffs, b=pi_max)
        root_z = functions.trans(root, **kwargs)
        args = (beta, pi_mu, pi_err, n)
        curve = functions.d2logIJ_dp2(root, *args, **kwargs)/functions.jac(root, **kwargs)**2
        I[i,:] = functions.integrate_gh(integrand,
                                                  functions.jac,
                                                  functions.trans_i,
                                                  root_z, 1/np.sqrt(-curve), args, kwargs)

    log_p = np.log(np.sum(I*weights[:,np.newaxis],axis=0))

    return logN + log_mnorm + log_pnorm + log_m + log_p

def integral_model(params, bins=None, fid_pars=None):

    N = np.exp(params[0])
    return N

def integral_model_SF(params, bins=None, fid_pars=None):

    N = np.exp(params[0])
    hz = 1/np.exp(params[1])
    alpha = params[2]

    Mmax = fid_pars['Mbol_max_true']
    theta= fid_pars['lat_min']
    mSF = fid_pars['m_sf']
    s_bound = 10**((mSF -(Mmax+10))/5)

    beta1 = np.sin(theta)/hz
    beta2 = 1/hz

    I = np.tan(theta)**2 / (hz**2) * ( \
                scipy.special.gamma(2)*scipy.special.gammainc(2,beta1*s_bound)/beta1**2 \
              - scipy.special.gamma(2)*scipy.special.gammainc(2,beta2*s_bound)/beta2**2 \
        + s_bound**alpha * (
                beta1**(alpha-2) * scipy.special.gamma(2-alpha)*\
                                   scipy.special.gammaincc(2-alpha,beta1*s_bound) \
              - beta2**(alpha-2) * scipy.special.gamma(2-alpha)*\
                                   scipy.special.gammaincc(2-alpha,beta2*s_bound) ))

    #print(I)

    return N * I


# Plotting modules
import matplotlib, matplotlib.pyplot as plt
font = {'family' : 'serif', 'weight' : 'normal',
        'size'   : 16}
legend = {'fontsize': 16}
matplotlib.rc('font', **font)
matplotlib.rc('legend', **legend)

def _test_perrmodel(sample_2d, params, model, fid_pars=None, rng=None,
                     idxs=None):

    if idxs is None: idxs = np.arange(sample_2d.shape[1])
    if rng is None: rng = [1e-10, 1-1e-10]
    roots, coeffs, kwargs, args = model(sample_2d[:,idxs], params, fid_pars=fid_pars, get_coeffs=True)
    beta, pi_mu, pi_err, n = args
    print(roots)

    x = np.zeros((len(idxs),1000))
    grad = np.zeros((len(idxs),1000))
    for j in range(len(idxs)):
        x[j] = np.linspace(rng[0]*kwargs['b'][j], rng[1]*kwargs['b'][j], 1000)
        grad[j] = [functions.polynomial_jit(x[j,i], coeffs[j]) for i in range(x.shape[1])]

    jac = np.array([functions.jac(x[:,i], **kwargs) for i in range(x.shape[1])]).T
    curve = np.array([functions.d2logIJ_dp2(x[:,i], *args, **kwargs) for i in range(x.shape[1])]).T / jac**2
    I = integrand(x[np.newaxis,:], beta[:,np.newaxis], pi_mu[:,np.newaxis], pi_err[:,np.newaxis],
                              n, b=kwargs['b'])[0]
    IJ = I/jac


    fig, axes = plt.subplots(len(idxs), 1, figsize=(8,len(idxs)*4))
    for j in range(len(idxs)):
        plt.sca(axes[j])
        plt.plot(x[j], I[j]/np.max(I[j]), label='integrand')
        plt.plot(x[j], IJ[j]/np.max(np.abs(IJ[j])), label='IJ')
        plt.plot(x[j], grad[j]/np.max(np.abs(grad[j])), label='gradient')
        plt.plot(x[j], curve[j]/np.max(np.abs(curve[j])), label='curvature')
        plt.plot(x[j], 1/jac[j]/np.max(np.abs(1/jac[j])), label='jacobian')

        plt.plot([roots[j], roots[j]], [-1, 1], '--r')

        plt.plot([rng[0]*kwargs['b'][j], rng[1]*kwargs['b'][j]], [0,0], '--k', alpha=0.5)

        plt.xlim(rng[0]*kwargs['b'][j], rng[1]*kwargs['b'][j])

    plt.sca(axes[0])
    plt.legend(bbox_to_anchor=(1.1,0.9))
