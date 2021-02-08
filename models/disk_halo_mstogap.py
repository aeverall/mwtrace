"""
disk_cone_plcut: Tracer Density model module.

- Exponential disk profile.
- Observing high latitudes with a projected cone.
- Power-law absolute magnitude distribution:
    - Positive power.
    - High (dim) cut-off.
"""

import sys, os
sys.path.append('../utilities/')
import functions
import numpy as np, scipy, healpy as hp
from numba import njit
from copy import deepcopy as copy


def logmodel_grad(sample, params, gmm=None, fid_pars=None, grad=False):

    # Observables
    pi_mu, log_pi_mu, abs_sin_lat, log_cos_lat, m_mu = sample

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    # Defined paramers
    theta=fid_pars['lat_min']; Mx=fid_pars['Mmax']; R0=fid_pars['R0']

    # Absmag
    M = m_mu - 10 + 5*log_pi_mu/np.log(10)

    logcmpts = np.zeros((len(pi_mu), ncomponents))
    if grad: grad_lambda = np.zeros((7, len(pi_mu), ncomponents))
    weights = np.zeros(ncomponents)
    for j in range(ncomponents):
        output = fid_pars['models'][j](pi_mu, abs_sin_lat, m_mu, M, log_pi_mu,
                                                hz=transformed_params[j]['hz'],
                                                alpha1=transformed_params[j]['alpha1'],
                                                alpha2=transformed_params[j]['alpha2'],
                                                alpha3=transformed_params[j]['alpha3'],
                                                fD=transformed_params[j]['fD'],
                                                Mms=transformed_params[j]['Mms'],
                                                Mms1=transformed_params[j]['Mms1'],
                                                Mms2=transformed_params[j]['Mms2'],
                                                Mto=transformed_params[j]['Mto'],
                                                Mx=Mx, R0=R0, grad=grad)
        if grad: logcmpts[:,j], grad_lambda[:-1,:,j] = output
        else: logcmpts[:,j] = output

        weights[j] = transformed_params[j]['w']

    logsumexp = scipy.special.logsumexp(logcmpts, b=weights, axis=1)

    if not grad: return logsumexp + 2*np.log(np.tan(theta)) + log_cos_lat

    grad_model = np.zeros((len(params), len(pi_mu)))
    params_i = 0
    for j in range(ncomponents):
        reweight = (weights * np.exp(logcmpts)).T/np.exp(logsumexp)
        for par in fid_pars['free_pars'][j]:
            if par=='hz': grad_model[params_i] = grad_lambda[0,:,j].copy()*reweight[j]
            if par=='alpha3': grad_model[params_i] = grad_lambda[1,:,j].copy()*reweight[j]
            if par=='fD': grad_model[params_i] = grad_lambda[2,:,j].copy()*reweight[j]
            if par=='Mto': grad_model[params_i] = grad_lambda[3,:,j].copy()*reweight[j]
            if par=='w': grad_model[params_i] = np.exp(logcmpts[:,j])/np.exp(logsumexp)
            params_i+=1
    for par in fid_pars['free_pars']['shd']:
        if par=='alpha1': grad_model[params_i] = np.sum(grad_lambda[4,:,:] * weights * np.exp(logcmpts), axis=1)/np.exp(logsumexp)
        if par=='alpha2': grad_model[params_i] = np.sum(grad_lambda[5,:,:] * weights * np.exp(logcmpts), axis=1)/np.exp(logsumexp)
        params_i+=1

    jacobian = jacobian_params(params, fid_pars, ncomponents=ncomponents)

    return logsumexp, (grad_model.T*jacobian).T# + 2*np.log(np.tan(theta)) + log_cos_lat, grad_model

def log_expmodel_grad(pi_mu, abs_sin_lat, m_mu, M, log_pi_mu, hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, grad=False):

    ep1=1.3; ep2=2.3;
    a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    # Latent variables
    n1 = -(4 + alpha1*5/np.log(10))
    ng = -(4 + alphag*5/np.log(10))
    n2 = -(4 + alpha2*5/np.log(10))
    n3 = -(4 + alpha3*5/np.log(10))

    pop1 = M>Mms1
    popg = M>Mms2
    pop2 = M>Mto

    exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    b = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    log_Ams = np.log( fD*a1*a2 ) - scipy.special.logsumexp(exponent,b=b)

    log_AG = np.log(-alpha3) + np.log(1-fD)
    log_m = np.where(pop1, log_Ams - np.log(a1) + alpha1*(Mms+10-m_mu),
            np.where(popg, log_Ams + np.log(Ag) + alphag*(Mms+10-m_mu),
            np.where(pop2, log_Ams - np.log(a2) + alpha2*(Mms+10-m_mu),
                           log_AG  + alpha3*(Mto+10-m_mu))))

    log_pnorm = -3*np.log(hz)
    log_p = - abs_sin_lat/(hz*pi_mu) +  np.where(pop1, n1*log_pi_mu,
                                        np.where(popg, ng*log_pi_mu,
                                        np.where(pop2, n2*log_pi_mu,
                                                       n3*log_pi_mu)))

    log_lambda = log_pnorm + log_p + log_m

    if not grad: return log_lambda

    grad_lambda = np.zeros((log_pi_mu.shape[0], 6)) + np.nan
    # hz
    grad_lambda[:,0] = -3/hz + abs_sin_lat/(hz**2 * pi_mu)
    # fD
    grad_lambda[:,2] = np.where(pop2, 1/fD, -1/(1-fD))
    # Mto
    grad_lambda[:,3] = np.where(pop2, 1/(np.exp(np.log( fD*a1*a2 ) - log_Ams)) * (-a1*alpha2/a2 * np.exp(alpha2*(Mms-Mto))), alpha3)
    # alpha3
    grad_lambda[:,1] = np.where(pop2, 0, 1/alpha3 + Mto+10-m_mu - 5*log_pi_mu/np.log(10))

    b_alpha1 = np.array([a2/alpha1 * ((Mms-Mms1)-1/alpha1),
                        -a2/alpha1 * ((Mms-Mx)  -1/alpha1),
                        a2/alphag * (-1/alpha1 + 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                       -a2/alphag * ((Mms-Mms1)+ 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                       -a1/(alpha1*alpha2),
                        a1/(alpha1*alpha2)])
    b_alpha2 = np.array([-a2/(alpha1*alpha2),
                         a2/(alpha1*alpha2),
                         a2/alphag * ((Mms-Mms2)- 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                         a2/alphag * ( 1/alpha2 + 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                         a1/alpha2 * ((Mms-Mto) - 1/alpha2),
                        -a1/alpha2 * ((Mms-Mms2)- 1/alpha2)])
    # alpha1
    dlnAms_dalpha1 = -1/alpha1 - np.sum(b_alpha1*np.exp(exponent))/np.sum(b*np.exp(exponent))
    grad_lambda[:,4] = np.where(pop1, dlnAms_dalpha1+1/alpha1+(Mms+10-m_mu) - 5/np.log(10)*log_pi_mu,
                       np.where(popg, dlnAms_dalpha1+(1/alpha1 + (Mms-Mms1))*(Mms-Mms2)/(Mms1-Mms2) - (1/alpha1 + (Mms-Mms1))/(Mms1-Mms2) * (Mms+10-m_mu - 5/np.log(10)*log_pi_mu),
                       np.where(pop2, dlnAms_dalpha1, 0)))

    # alpha2
    dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(exponent))/np.sum(b*np.exp(exponent))
    grad_lambda[:,5] = np.where(pop1, dlnAms_dalpha2,
                       np.where(popg, dlnAms_dalpha2-(1/alpha2 + (Mms-Mms2))*(Mms-Mms1)/(Mms1-Mms2) + (1/alpha2 + (Mms-Mms2))/(Mms1-Mms2) * (Mms+10-m_mu - 5/np.log(10)*log_pi_mu),
                       np.where(pop2, dlnAms_dalpha2+1/alpha2+(Mms+10-m_mu) - 5/np.log(10)*log_pi_mu, 0)))

    return log_lambda, grad_lambda.T#np.exp(log_lambda)*grad_lambda.T

def log_halomodel_grad(pi_mu, abs_sin_lat, m_mu, M, log_pi_mu, hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, grad=False):

    ep1=1.3; ep2=2.3;
    a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    # Latent variables
    n1 = -(4 + alpha1*5/np.log(10))
    ng = -(4 + alphag*5/np.log(10))
    n2 = -(4 + alpha2*5/np.log(10))
    n3 = -(4 + alpha3*5/np.log(10))

    pop1 = M>Mms1
    popg = M>Mms2
    pop2 = M>Mto

    exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    b = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    log_Ams = np.log( fD*a1*a2 ) - scipy.special.logsumexp(exponent,b=b)

    log_AG = np.log(-alpha3) + np.log(1-fD)

    log_m = np.where(pop1, log_Ams - np.log(a1) + alpha1*(Mms+10-m_mu),
            np.where(popg, log_Ams + np.log(Ag) + alphag*(Mms+10-m_mu),
            np.where(pop2, log_Ams - np.log(a2) + alpha2*(Mms+10-m_mu),
                           log_AG  + alpha3*(Mto+10-m_mu))))

    #pnorm = (8*scipy.special.gamma(hz/2))/(R0**3 * np.sqrt(np.pi) * scipy.special.gamma((hz-3)/2))
    log_pnorm = np.log(8) + scipy.special.gammaln(hz/2) - 3*np.log(R0) - 0.5*np.log(np.pi) - scipy.special.gammaln((hz-3)/2)
    log_p = - hz/2 * np.log((abs_sin_lat**2)/(pi_mu**2 * R0**2) + 1)  + np.where(pop1, n1*log_pi_mu,
                                                                        np.where(popg, ng*log_pi_mu,
                                                                        np.where(pop2, n2*log_pi_mu,
                                                                                       n3*log_pi_mu)))

    log_lambda = log_m + log_pnorm + log_p
    if not grad: return log_lambda

    grad_lambda = np.zeros((log_pi_mu.shape[0], 6)) + np.nan
    # hz
    grad_lambda[:,0] = scipy.special.digamma(hz/2)/2 - scipy.special.digamma((hz-3)/2)/2 - 1/2 * np.log((abs_sin_lat**2)/(pi_mu**2 * R0**2) + 1)
    # alpha3
    grad_lambda[:,1] = np.where(pop2, 0, 1/alpha3 + Mto+10-m_mu - 5*log_pi_mu/np.log(10))
    # fD
    grad_lambda[:,2] = np.where(pop2, 1/fD, -1/(1-fD))
    # Mto
    grad_lambda[:,3] = np.where(pop2, 1/(np.exp(np.log( fD*a1*a2 ) - log_Ams)) * (-a1*alpha2/a2 * np.exp(alpha2*(Mms-Mto))), alpha3)

    b_alpha1 = np.array([a2/alpha1 * ((Mms-Mms1)-1/alpha1),
                        -a2/alpha1 * ((Mms-Mx)  -1/alpha1),
                        a2/alphag * (-1/alpha1 + 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                       -a2/alphag * ((Mms-Mms1)+ 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                       -a1/(alpha1*alpha2),
                        a1/(alpha1*alpha2)])
    b_alpha2 = np.array([-a2/(alpha1*alpha2),
                         a2/(alpha1*alpha2),
                         a2/alphag * ((Mms-Mms2)- 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                         a2/alphag * ( 1/alpha2 + 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                         a1/alpha2 * ((Mms-Mto) - 1/alpha2),
                        -a1/alpha2 * ((Mms-Mms2)- 1/alpha2)])
    # alpha1
    dlnAms_dalpha1 = -1/alpha1 - np.sum(b_alpha1*np.exp(exponent))/np.sum(b*np.exp(exponent))
    grad_lambda[:,4] = np.where(pop1, dlnAms_dalpha1+1/alpha1+(Mms+10-m_mu) - 5/np.log(10)*log_pi_mu,
                       np.where(popg, dlnAms_dalpha1+(1/alpha1 + (Mms-Mms1))*(Mms-Mms2)/(Mms1-Mms2) - (1/alpha1 + (Mms-Mms1))/(Mms1-Mms2) * (Mms+10-m_mu - 5/np.log(10)*log_pi_mu),
                       np.where(pop2, dlnAms_dalpha1, 0)))

    # alpha2
    dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(exponent))/np.sum(b*np.exp(exponent))
    grad_lambda[:,5] = np.where(pop1, dlnAms_dalpha2,
                       np.where(popg, dlnAms_dalpha2-(1/alpha2 + (Mms-Mms2))*(Mms-Mms1)/(Mms1-Mms2) + (1/alpha2 + (Mms-Mms2))/(Mms1-Mms2) * (Mms+10-m_mu - 5/np.log(10)*log_pi_mu),
                       np.where(pop2, dlnAms_dalpha2+1/alpha2+(Mms+10-m_mu) - 5/np.log(10)*log_pi_mu, 0)))

    return log_lambda, grad_lambda.T#np.exp(log_lambda)*grad_lambda.T

def logmodel_perr(sample, params, gmm=None, fid_pars=None, grad=False):

    # Observables
    pi_mu, pi_err, abs_sin_lat, log_cos_lat, m_mu, log_pi_err = sample

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    # Defined paramers
    theta=fid_pars['lat_min']; Mx=fid_pars['Mmax']; R0=fid_pars['R0']

    logcmpts = np.zeros((len(pi_mu), ncomponents))
    weights = np.zeros(ncomponents)
    for j in range(ncomponents):
        logcmpts[:,j] = fid_pars['models'][j](pi_mu, pi_err, abs_sin_lat, m_mu, log_pi_err,
                                                hz=transformed_params[j]['hz'],
                                                alpha1=transformed_params[j]['alpha1'],
                                                alpha2=transformed_params[j]['alpha2'],
                                                alpha3=transformed_params[j]['alpha3'],
                                                fD=transformed_params[j]['fD'],
                                                Mms=transformed_params[j]['Mms'],
                                                Mms1=transformed_params[j]['Mms1'],
                                                Mms2=transformed_params[j]['Mms2'],
                                                Mto=transformed_params[j]['Mto'],
                                                Mx=Mx, R0=R0)

        weights[j] = transformed_params[j]['w']

    logsumexp = scipy.special.logsumexp(logcmpts, b=weights, axis=1) + log_cos_lat

    return 2*np.log(np.tan(theta)) + logsumexp

def log_expmodel_perr(pi_mu, pi_err, abs_sin_lat, m_mu, log_pi_err, hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, degree=21):

    ep1=1.3; ep2=2.3;
    a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    beta = abs_sin_lat/hz

    # Latent variables
    n1 = -(4 + alpha1*5/np.log(10))
    ng = -(4 + alphag*5/np.log(10))
    n2 = -(4 + alpha2*5/np.log(10))
    n3 = -(4 + alpha3*5/np.log(10))

    log_Ams = np.log( fD*a1*a2 ) - \
              scipy.special.logsumexp(np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                                                alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                                                alpha2*(Mms-Mto), alpha2*(Mms-Mms2)]),
                                    b=np.array([a2/alpha1, -a2/alpha1,
                                                a2/alphag, -a2/alphag,
                                                a1/alpha2, -a1/alpha2]))
    log_AG = np.log(-alpha3) + np.log(1-fD)

    log_pnorm = -3*np.log(hz)

    # Absolute magnitude not known
    Mag_bounds = [-np.inf, Mto, Mms2, Mms1, Mx]
    Mag_n =      [n3,  n2,   ng,   n1]
    Mag_norm =   [log_AG  + alpha3*(Mto+10-m_mu),
                  log_Ams - np.log(a2) + alpha2*(Mms+10-m_mu),
                  log_Ams + np.log(Ag) + alphag*(Mms+10-m_mu),
                  log_Ams - np.log(a1) + alpha1*(Mms+10-m_mu)]

    p_model = np.zeros((4, len(pi_mu)))
    for ii in range(4): #THIS NEEDS TO BE REVERTED BACK TO 4.

        p_integral = np.zeros(len(pi_mu))

        p_min = np.exp((Mag_bounds[ii  ]+10-m_mu)*np.log(10)/5)
        p_max = np.exp((Mag_bounds[ii+1]+10-m_mu)*np.log(10)/5)

        n = Mag_n[ii]
        args = (abs_sin_lat/hz, n*np.ones(len(pi_mu)), pi_mu, pi_err)
        grad_min = expmodel_perr_grad(p_min, args)
        grad_max = expmodel_perr_grad(p_max, args)
        legendre = grad_min*grad_max>0
        legendre[:] = False

        # Gauss - Legendre Quadrature
        a = p_min[legendre][:,np.newaxis]; b = p_max[legendre][:,np.newaxis]
        args = (beta[legendre], n*np.ones(len(pi_mu[legendre])), pi_mu[legendre], pi_err[legendre])
        leg_nodes, leg_weights = np.polynomial.legendre.leggauss(degree)
        leg_nodes = leg_nodes[np.newaxis,:] * (b-a)/2 + (b+a)/2
        p_integral[legendre] = np.sum(expmodel_perr_integrand(leg_nodes.T, *args), axis=0)

        # Gauss - Hermite Quadrature
        a = p_min[~legendre]; b = p_max[~legendre]

        args = (beta[~legendre], n*np.ones(len(pi_mu[~legendre])), pi_mu[~legendre], pi_err[~legendre], a, b)
        p_mode = functions.get_fooroots_ridder_hm(expmodel_perr_logit_grad, a=a+1e-15, b=b, args=args)

        curve = expmodel_perr_d2logIJ_dp2(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                    functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2
        z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)

        sigma = 1/np.sqrt(-curve)
        p_integral[~legendre] = functions.integrate_gh_gap(expmodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=10)

        p_model[ii] = p_integral

    log_p = scipy.special.logsumexp(Mag_norm, b=p_model, axis=0)

    return log_pnorm + log_p - 0.5*np.log(2*np.pi) - log_pi_err


def logmodel_perr_jitted(sample, params, gmm=None, fid_pars=None, grad=False):

    # Observables
    pi_mu, pi_err, abs_sin_lat, log_cos_lat, m_mu, log_pi_err = sample

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)


    logcmpts = np.zeros((len(pi_mu), ncomponents))
    weights = np.zeros(ncomponents)
    p_model = np.zeros((4,len(pi_mu)))
    for j in range(ncomponents):

        alpha1 = transformed_params[j]['alpha1']
        alpha2 = transformed_params[j]['alpha2']
        alpha3 = transformed_params[j]['alpha3']
        Mms = transformed_params[j]['Mms']
        Mms1 = transformed_params[j]['Mms1']
        Mms2 = transformed_params[j]['Mms2']
        Mto = transformed_params[j]['Mto']
        fD = transformed_params[j]['fD']
        hz = transformed_params[j]['hz']

        # Defined paramers
        theta=fid_pars['lat_min']; Mx=fid_pars['Mmax']; R0=fid_pars['R0']
        beta = abs_sin_lat/hz

        ep1=1.3; ep2=2.3;
        a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);
        alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
        Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))
        # Latent variables
        n1 = -(4 + alpha1*5/np.log(10))
        ng = -(4 + alphag*5/np.log(10))
        n2 = -(4 + alpha2*5/np.log(10))
        n3 = -(4 + alpha3*5/np.log(10))
        log_Ams = np.log( fD*a1*a2 ) - \
                  scipy.special.logsumexp(np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                                                    alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                                                    alpha2*(Mms-Mto), alpha2*(Mms-Mms2)]),
                                        b=np.array([a2/alpha1, -a2/alpha1,
                                                    a2/alphag, -a2/alphag,
                                                    a1/alpha2, -a1/alpha2]))
        log_AG = np.log(-alpha3) + np.log(1-fD)
        # Absolute magnitude not known
        Mag_bounds = [-np.inf, Mto, Mms2, Mms1, Mx]
        Mag_n =      [n3,  n2,   ng,   n1]
        Mag_norm =   [log_AG  + alpha3*(Mto+10-m_mu),
                      log_Ams - np.log(a2) + alpha2*(Mms+10-m_mu),
                      log_Ams + np.log(Ag) + alphag*(Mms+10-m_mu),
                      log_Ams - np.log(a1) + alpha1*(Mms+10-m_mu)]

        log_pnorm = np.log(8) + scipy.special.gammaln(hz/2) - 3*np.log(R0) - 0.5*np.log(np.pi) - scipy.special.gammaln((hz-3)/2)
        log_pnorm = -3*np.log(hz)

        p_model = fid_pars['models'][j](p_model, beta, pi_mu, pi_err, m_mu, Mag_bounds, Mag_n)
        logcmpts[:,j] = scipy.special.logsumexp(Mag_norm, b=p_model, axis=0) + log_pnorm - 0.5*np.log(2*np.pi) - log_pi_err

        weights[j] = transformed_params[j]['w']

    logsumexp = scipy.special.logsumexp(logcmpts, b=weights, axis=1) + log_cos_lat

    return 2*np.log(np.tan(theta)) + logsumexp

def log_expmodel_perr_jitted(p_model, beta, pi_mu, pi_err, m_mu, Mag_bounds, Mag_n, degree=21):

    for ii in range(1): #THIS NEEDS TO BE REVERTED BACK TO 4.

        p_integral = np.zeros(len(pi_mu))

        p_min = np.exp((Mag_bounds[ii  ]+10-m_mu)*np.log(10)/5)
        p_max = np.exp((Mag_bounds[ii+1]+10-m_mu)*np.log(10)/5)

        n = Mag_n[ii]
        args = (beta, n*np.ones(len(pi_mu)), pi_mu, pi_err)
        grad_min = expmodel_perr_grad(p_min, args)
        grad_max = expmodel_perr_grad(p_max, args)
        legendre = grad_min*grad_max>0
        legendre[:] = False

        # Gauss - Legendre Quadrature
        a = p_min[legendre][:,np.newaxis]; b = p_max[legendre][:,np.newaxis]
        args = (beta[legendre], n*np.ones(len(pi_mu[legendre])), pi_mu[legendre], pi_err[legendre])
        leg_nodes, leg_weights = np.polynomial.legendre.leggauss(degree)
        leg_nodes = leg_nodes[np.newaxis,:] * (b-a)/2 + (b+a)/2
        p_integral[legendre] = np.sum(expmodel_perr_integrand(leg_nodes.T, *args), axis=0)

        # Gauss - Hermite Quadrature
        a = p_min[~legendre]; b = p_max[~legendre]

        args = (beta[~legendre], n*np.ones(len(pi_mu[~legendre])), pi_mu[~legendre], pi_err[~legendre], p_max)
        p_mode = functions.get_fooroots_ridder_hm(expmodel_perr_logit_grad, a=a, b=b, args=args)

        curve = expmodel_perr_d2logIJ_dp2(p_mode, *args[:-1], transform='logit_ab', a=a, b=b) / \
                                    functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2
        z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)

        sigma = 1/np.sqrt(-curve)
        p_integral[~legendre] = functions.integrate_gh_gap(expmodel_perr_integrand, z_mode, sigma, args[:-1], transform='logit_ab', a=a, b=b, degree=10)

        p_model[ii] = p_integral

    return p_model

#@njit
def expmodel_perr_grad(p, args):
    beta, n, mu, err = args
    return p**3 - mu*p**2 - n*err**2*p - beta*err**2

# @njit
# def expmodel_perr_logit_grad(p, args):
#     beta, n, mu, err, pmax = args
#     return p**4 - (pmax+mu)*p**3 + (pmax*mu - (n+2)*err**2)*p**2 + ( ((n+1)*pmax-beta) * err**2 )*p + beta*pmax*err**2
@njit
def expmodel_perr_logit_grad(p, args):
    beta, n, mu, err, a, b = args
    return p**2 * (a+b-2*p) \
         + (n*p + beta - p**2*(p-mu)/err**2) * (p-a)*(b-p)
#@njit
def expmodel_perr_integrand(p, beta, n, mu, err):
    #beta, n, h, mu, err = args
    return p**n * np.exp(-beta/p - (p-mu)**2/(2*err**2))
#@njit
def expmodel_integrand(p, beta, n, h, mu, err):
    #beta, n, h, mu, err = args
    return p**n * np.exp(-beta/p)
@njit
def expmodel_perr_d2logIJ_dp2(p, beta, n, mu, err, transform='none', b=None, a=None):
    d2logI_dp2 = -n/p**2 - 2*beta/p**3 - 1/err**2
    if   transform=='none':     return d2logI_dp2
    elif transform=='logit':    return d2logI_dp2 - 1/p**2 - 1/(p-b)**2
    elif transform=='logit_ab': return d2logI_dp2 - 1/(p-a)**2 - 1/(p-b)**2

def log_halomodel_perr(pi_mu, pi_err, abs_sin_lat, m_mu, log_pi_err, hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, degree=21):

    ep1=1.3; ep2=2.3;
    a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    # Latent variables
    n1 = -(4 + alpha1*5/np.log(10))
    ng = -(4 + alphag*5/np.log(10))
    n2 = -(4 + alpha2*5/np.log(10))
    n3 = -(4 + alpha3*5/np.log(10))

    exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    b = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    log_Ams = np.log( fD*a1*a2 ) - scipy.special.logsumexp(exponent,b=b)

    log_AG = np.log(-alpha3) + np.log(1-fD)

    beta = abs_sin_lat/R0
    log_pnorm = np.log(8) + scipy.special.gammaln(hz/2) - 3*np.log(R0) - 0.5*np.log(np.pi) - scipy.special.gammaln((hz-3)/2)

    # Absolute magnitude not known
    Mag_bounds = [-np.inf, Mto, Mms2, Mms1, Mx]
    Mag_n =      [n3,  n2,   ng,   n1]
    Mag_norm =   [log_AG  + alpha3*(Mto+10-m_mu),
                  log_Ams - np.log(a2) + alpha2*(Mms+10-m_mu),
                  log_Ams + np.log(Ag) + alphag*(Mms+10-m_mu),
                  log_Ams - np.log(a1) + alpha1*(Mms+10-m_mu)]

    p_model = np.zeros((4, len(pi_mu)))
    for ii in range(4):

        p_integral = np.zeros(len(pi_mu))

        n = Mag_n[ii]
        p_min = np.exp((Mag_bounds[ii  ]+10-m_mu)*np.log(10)/5)
        p_max = np.exp((Mag_bounds[ii+1]+10-m_mu)*np.log(10)/5)

        args = (beta, n*np.ones(len(pi_mu)), hz*np.ones(len(pi_mu)), pi_mu, pi_err)
        grad_min = halomodel_perr_grad(p_min, *args)
        grad_max = halomodel_perr_grad(p_max, *args)
        legendre = grad_min*grad_max>0
        legendre[:]=False

        # Gauss - Legendre Quadrature
        a = p_min[legendre]; b = p_max[legendre]
        args = (beta[legendre], n*np.ones(len(pi_mu))[legendre],
                    hz*np.ones(len(pi_mu))[legendre], pi_mu[legendre], pi_err[legendre])
        leg_nodes, leg_weights = np.polynomial.legendre.leggauss(degree)
        leg_nodes = leg_nodes[np.newaxis,:].T * (b-a)/2 + (b+a)/2
        p_integral[legendre] = np.sum(halomodel_perr_integrand(leg_nodes, *args), axis=0)

        # Gauss - Hermite Quadrature
        a = p_min[~legendre]; b = p_max[~legendre]
        args = (beta[~legendre], n*np.ones(len(pi_mu[~legendre])),
                    hz*np.ones(len(pi_mu[~legendre])), pi_mu[~legendre], pi_err[~legendre], a, b)
        p_mode = functions.get_fooroots_ridder_hm(halomodel_perr_logit_grad, a=a+1e-15, b=b, args=args)
        curve = halomodel_perr_d2logIJ_dp2(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                    functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2

        z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
        sigma = 1/np.sqrt(-curve)
        p_integral[~legendre] = functions.integrate_gh_gap(halomodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=10)

        p_model[ii] = p_integral

    log_p = scipy.special.logsumexp(Mag_norm, b=p_model, axis=0)
    #print(log_p)

    return log_pnorm + log_p - 0.5*np.log(2*np.pi) - log_pi_err

@njit
def halomodel_perr_grad(p, beta, n, h, mu, err):
    return -p**4 + mu*p**3 - (beta**2-n*err**2)*p**2 + mu*beta**2*p + (n+h)*err**2*beta**2
@njit
def halomodel_perr_logit_grad(p, args):
    beta, n, h, mu, err, pmin, pmax = args
    return p**5 - (pmax+mu)*p**4 \
           + ((beta**2+mu*pmax) - (n+2)*err**2)*p**3 \
           + ((n+1)*pmax*err**2 - beta**2*(mu+pmax))*p**2 \
           + (mu*beta**2*pmax - (n+2+h)*beta**2*err**2)*p + ((h+n+1)*pmax)*beta**2*err**2
@njit
def halomodel_perr_logit_grad(p, args):
    beta, n, h, mu, err, a, b = args
    return p*(beta**2 + p**2) * (a+b-2*p) +\
          ((n - p*(p-mu)/err**2)*(beta**2+p**2) + h*beta**2) * (p-a)*(b-p)
@njit
def halomodel_perr_integrand(p, beta, n, h, mu, err):
    return p**n * (beta**2/p**2 + 1)**(-h/2) * np.exp(-(p-mu)**2/(2*err**2))

@njit
def halomodel_perr_d2logIJ_dp2(p, beta, n, h, mu, err, transform='none', b=None, a=None):
    d2logI_dp2 = -(n+h)/p**2 + h*(p**2-beta**2)/(p**2+beta**2)**2 - 1/err**2
    if   transform=='none':     return d2logI_dp2
    elif transform=='logit':    return d2logI_dp2 - 1/p**2 - 1/(p-b)**2
    elif transform=='logit_ab': return d2logI_dp2 - 1/(p-a)**2 - 1/(p-b)**2


#%% Integral of model over parameter space.
def integral_model(params, bins=None, fid_pars=None, grad=False):

    transformed_params = combined_params(params, fid_pars, ncomponents=fid_pars['ncomponents'])
    N=0.
    for j in range(fid_pars['ncomponents']): N+=transformed_params[j]['w']

    if not grad: return N
    elif grad:
        integral_grad = np.zeros(len(params))

        params_i=0
        for j in range(fid_pars['ncomponents']):
            for par in fid_pars['free_pars'][j]:
                if par=='w': integral_grad[params_i]=1
                params_i += 1

        jacobian = jacobian_params(params, fid_pars, ncomponents=fid_pars['ncomponents'])
        return N, integral_grad*jacobian


#%% Parameter Transformations
def combined_params(params, fid_pars, ncomponents=1, transform=True):

    # ['pi', 'hz','alpha1','alpha2','alpha3','fD','Mms','Mto'] must all be in free or fixed pars for all components
    free_pars = fid_pars['free_pars']
    fixed_pars = fid_pars['fixed_pars']
    functions = fid_pars['functions']
    output_pars = {}

    params_i = 0
    for j in range(ncomponents):
        output_pars[j]={}
        for par in fid_pars['free_pars'][j]:
            if transform: output_pars[j][par]=fid_pars['functions'][j][par](params[params_i]); params_i += 1;
            else: output_pars[j][par]=params[params_i]; params_i += 1;
        for par in fid_pars['fixed_pars'][j].keys():
            output_pars[j][par]=fid_pars['fixed_pars'][j][par]
    for par in fid_pars['free_pars']['shd']:
        for j in range(ncomponents):
            if transform: output_pars[j][par]=fid_pars['functions']['shd'][par](params[params_i])
            else: output_pars[j][par]=params[params_i]
        params_i += 1

    return output_pars

def jacobian_params(params, fid_pars, ncomponents=1, transform=True):

    # ['pi', 'hz','alpha1','alpha2','alpha3','fD','Mms','Mto'] must all be in free or fixed pars for all components
    free_pars = fid_pars['free_pars']
    fixed_pars = fid_pars['fixed_pars']

    jacobian = np.ones(len(params))

    params_i = 0
    for j in range(ncomponents):
        for par in fid_pars['free_pars'][j]:
            if transform: jacobian[params_i]=fid_pars['jacobians'][j][par](fid_pars['functions'][j][par](params[params_i])); params_i += 1;
            else: params_i += 1;
    for par in fid_pars['free_pars']['shd']:
        for j in range(ncomponents):
            if transform: jacobian[params_i]=fid_pars['jacobians'][j][par](fid_pars['functions'][j][par](params[params_i]))
            else: jacobian[params_i]=params[params_i]
        params_i += 1

    return jacobian


#%% Unit Tests
import unittest, tqdm

class TestPoissonBinomial(unittest.TestCase):

    def __init__(self,*args,**kwargs):
        super(TestPoissonBinomial, self).__init__(*args, **kwargs)


    def test_halo_grad_perr(self):

        test_args = (1,3,2.,0.5,0.1,0.,1.)
        grad = lambda x: halomodel_perr_grad(x, *test_args[:-1])
        model = lambda x: halomodel_perr_integrand(x, *test_args[:-1])
        self.assertAlmostEqual( grad(0.01), scipy.optimize.approx_fprime(np.array([0.01]), model, 1e-12), 8)

    def test_halo_logit_grad_perr(self):

        test_args = (1,3,2.,0.5,0.1,0.,0.,1.)
        beta, n, h, mu, err, pmax = test_args
        grad = lambda x: dh_msto.halomodel_perr_logit_grad(x, test_args) \
                       * dh_msto.halomodel_perr_integrand(x, *test_args[:-2])/(x*(beta**2+x**2))
        model = lambda x: (x-test_args[-2])*(test_args[-1]-x)*dh_msto.halomodel_perr_integrand(x, *test_args[:-2])

        self.assertAlmostEqual( grad(0.01), scipy.optimize.approx_fprime(np.array([0.01]), model, 1e-12), 8)
        self.assertAlmostEqual( grad(0.99), scipy.optimize.approx_fprime(np.array([0.99]), model, 1e-12), 8)

        test_args = (0.11459087188687923, -1.9315154043531053, 62.460138858539736,
                     -0.6133993246056172,  1.002678948333817, 0.2, 0.5709469567273955)
        beta, n, h, mu, err, pmax = test_args
        grad = lambda x: dh_msto.halomodel_perr_logit_grad(x, test_args) \
                       * dh_msto.halomodel_perr_integrand(x, *test_args[:-2])/(x*(beta**2+x**2))
        model = lambda x: (x-test_args[-2])*(test_args[-1]-x)*dh_msto.halomodel_perr_integrand(x, *test_args[:-2])
        self.assertAlmostEqual( grad(0.57), scipy.optimize.approx_fprime(np.array([0.57]), model, 1e-12), 8)
        self.assertAlmostEqual( grad(0.01), scipy.optimize.approx_fprime(np.array([0.01]), model, 1e-12), 8)

    def test_disk_logit_grad_perr(self):

        test_args = (1,3,2.,0.5,0.1,0.,0.,1.)
        beta, n, mu, err, a, b = test_args
        grad = lambda x: dh_msto.expmodel_perr_logit_grad(x, test_args) \
                       * dh_msto.expmodel_perr_integrand(x, *test_args[:-2])/x**2
        model = lambda x: (x-test_args[-2])*(test_args[-1]-x)*dh_msto.expmodel_perr_integrand(x, *test_args[:-2])

        self.assertAlmostEqual( grad(0.01), scipy.optimize.approx_fprime(np.array([0.01]), model, 1e-12), 8)
        self.assertAlmostEqual( grad(0.99), scipy.optimize.approx_fprime(np.array([0.99]), model, 1e-12), 8)

        test_args = (0.11459087188687923, -1.9315154043531053, 62.460138858539736,
                     -0.6133993246056172,  1.002678948333817, 0.2, 0.5709469567273955)
        beta, n, mu, err, a, b = test_args
        grad = lambda x: dh_msto.expmodel_perr_logit_grad(x, test_args) \
                       * dh_msto.expmodel_perr_integrand(x, *test_args[:-2])/(x*(beta**2+x**2))
        model = lambda x: (x-test_args[-2])*(test_args[-1]-x)*dh_msto.expmodel_perr_integrand(x, *test_args[:-2])
        self.assertAlmostEqual( grad(0.57), scipy.optimize.approx_fprime(np.array([0.57]), model, 1e-12), 8)
        self.assertAlmostEqual( grad(0.01), scipy.optimize.approx_fprime(np.array([0.01]), model, 1e-12), 8)

if __name__ == '__main__':
    unittest.main()
