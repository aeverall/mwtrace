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
import functions, sf_utils, numba_special
import numpy as np, scipy, healpy as hp, h5py
from numba import njit
from copy import deepcopy as copy

ln10 = np.log(10)
eps = 1e-12

# Integration points used for Gaia SF integration
degree=21
nodes_leg, weights_leg = np.polynomial.legendre.leggauss(degree)
nodes_lag, weights_lag = np.polynomial.laguerre.laggauss(degree)


def logmodel_grad(sample, params, gmm=None, fid_pars=None, grad=False):

    # Observables
    pi_mu, log_pi_mu, abs_sin_lat, log_cos_lat, m_mu = sample

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    # Defined paramers
    theta=fid_pars['lat_min']; Mx=fid_pars['Mmax']; R0=fid_pars['R0']
    nu_norm=fid_pars['halomodel_nu_norm']

    # Absmag
    M = m_mu - 10 + 5*log_pi_mu/ln10

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
                                                Mx=Mx, R0=R0, grad=grad,
                                                nu_norm=nu_norm)
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
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, grad=False, nu_norm=None):

    ep1=1.3; ep2=2.3;
    a1=-ln10*(ep1-1)/(2.5*alpha1); a2=-ln10*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    # Latent variables
    n1 = -(4 + alpha1*5/ln10)
    ng = -(4 + alphag*5/ln10)
    n2 = -(4 + alpha2*5/ln10)
    n3 = -(4 + alpha3*5/ln10)

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
    grad_lambda[:,1] = np.where(pop2, 0, 1/alpha3 + Mto+10-m_mu - 5*log_pi_mu/ln10)

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
    grad_lambda[:,4] = np.where(pop1, dlnAms_dalpha1+1/alpha1+(Mms+10-m_mu) - 5/ln10*log_pi_mu,
                       np.where(popg, dlnAms_dalpha1+(1/alpha1 + (Mms-Mms1))*(Mms-Mms2)/(Mms1-Mms2) - (1/alpha1 + (Mms-Mms1))/(Mms1-Mms2) * (Mms+10-m_mu - 5/ln10*log_pi_mu),
                       np.where(pop2, dlnAms_dalpha1, 0)))

    # alpha2
    dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(exponent))/np.sum(b*np.exp(exponent))
    grad_lambda[:,5] = np.where(pop1, dlnAms_dalpha2,
                       np.where(popg, dlnAms_dalpha2-(1/alpha2 + (Mms-Mms2))*(Mms-Mms1)/(Mms1-Mms2) + (1/alpha2 + (Mms-Mms2))/(Mms1-Mms2) * (Mms+10-m_mu - 5/ln10*log_pi_mu),
                       np.where(pop2, dlnAms_dalpha2+1/alpha2+(Mms+10-m_mu) - 5/ln10*log_pi_mu, 0)))

    return log_lambda, grad_lambda.T#np.exp(log_lambda)*grad_lambda.T

def log_halomodel_grad(pi_mu, abs_sin_lat, m_mu, M, log_pi_mu, hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, grad=False, nu_norm=None):

    ep1=1.3; ep2=2.3;
    a1=-ln10*(ep1-1)/(2.5*alpha1); a2=-ln10*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    # Latent variables
    n1 = -(4 + alpha1*5/ln10)
    ng = -(4 + alphag*5/ln10)
    n2 = -(4 + alpha2*5/ln10)
    n3 = -(4 + alpha3*5/ln10)

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

    #log_pnorm = (8*scipy.special.gamma(hz/2))/(R0**3 * np.sqrt(np.pi) * scipy.special.gamma((hz-3)/2))
    # log_pnorm = np.log(8) + scipy.special.gammaln(hz/2) - 3*np.log(R0) - 0.5*np.log(np.pi) - scipy.special.gammaln((hz-3)/2)
    log_pnorm = nu_norm(hz)

    log_p = - hz/2 * np.log((abs_sin_lat**2)/(pi_mu**2 * R0**2) + 1)  + np.where(pop1, n1*log_pi_mu,
                                                                        np.where(popg, ng*log_pi_mu,
                                                                        np.where(pop2, n2*log_pi_mu,
                                                                                       n3*log_pi_mu)))

    log_lambda = log_m + log_pnorm + log_p
    if not grad: return log_lambda

    grad_lambda = np.zeros((log_pi_mu.shape[0], 6)) + np.nan
    # hz
    # grad_lambda[:,0] = scipy.special.digamma(hz/2)/2 - scipy.special.digamma((hz-3)/2)/2 - 1/2 * np.log((abs_sin_lat**2)/(pi_mu**2 * R0**2) + 1)
    dhz = hz*1e-5
    grad_lambda[:,0] = (nu_norm(hz+dhz)-log_pnorm)/dhz - 1/2 * np.log((abs_sin_lat**2)/(pi_mu**2 * R0**2) + 1)
    # alpha3
    grad_lambda[:,1] = np.where(pop2, 0, 1/alpha3 + Mto+10-m_mu - 5*log_pi_mu/ln10)
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
    grad_lambda[:,4] = np.where(pop1, dlnAms_dalpha1+1/alpha1+(Mms+10-m_mu) - 5/ln10*log_pi_mu,
                       np.where(popg, dlnAms_dalpha1+(1/alpha1 + (Mms-Mms1))*(Mms-Mms2)/(Mms1-Mms2) - (1/alpha1 + (Mms-Mms1))/(Mms1-Mms2) * (Mms+10-m_mu - 5/ln10*log_pi_mu),
                       np.where(pop2, dlnAms_dalpha1, 0)))

    # alpha2
    dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(exponent))/np.sum(b*np.exp(exponent))
    grad_lambda[:,5] = np.where(pop1, dlnAms_dalpha2,
                       np.where(popg, dlnAms_dalpha2-(1/alpha2 + (Mms-Mms2))*(Mms-Mms1)/(Mms1-Mms2) + (1/alpha2 + (Mms-Mms2))/(Mms1-Mms2) * (Mms+10-m_mu - 5/ln10*log_pi_mu),
                       np.where(pop2, dlnAms_dalpha2+1/alpha2+(Mms+10-m_mu) - 5/ln10*log_pi_mu, 0)))

    return log_lambda, grad_lambda.T#np.exp(log_lambda)*grad_lambda.T


def logmodel_perr_grad(sample, params, gmm=None, fid_pars=None, grad=False, degree=20):

    # Observables
    pi_mu, pi_err, abs_sin_lat, log_cos_lat, m_mu, log_pi_err = sample

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    # Defined paramers
    theta=fid_pars['lat_min']; Mx=fid_pars['Mmax']; R0=fid_pars['R0']; smax=fid_pars['smax']

    weights = np.zeros(ncomponents)
    logcmpts = np.zeros((len(pi_mu), ncomponents))
    if grad: grad_lambda = np.zeros((7, len(pi_mu), ncomponents))
    for j in range(ncomponents):
        output = fid_pars['models'][j](pi_mu, pi_err, abs_sin_lat, m_mu, log_pi_err,
                                                hz=transformed_params[j]['hz'],
                                                alpha1=transformed_params[j]['alpha1'],
                                                alpha2=transformed_params[j]['alpha2'],
                                                alpha3=transformed_params[j]['alpha3'],
                                                fD=transformed_params[j]['fD'],
                                                Mms=transformed_params[j]['Mms'],
                                                Mms1=transformed_params[j]['Mms1'],
                                                Mms2=transformed_params[j]['Mms2'],
                                                Mto=transformed_params[j]['Mto'],
                                                Mx=Mx, R0=R0, smax=smax, grad=grad, degree=degree,
                                                nu_norm=fid_pars['halomodel_nu_norm'])

        weights[j] = transformed_params[j]['w']
        if not grad:  logcmpts[:,j] = output
        else:         logcmpts[:,j], grad_lambda[:-1,:,j] = output
    #print(np.sum(grad_lambda, axis=1))

    logsumcmpts = scipy.special.logsumexp(logcmpts, b=weights, axis=1)

    if not grad: return logsumcmpts + 2*np.log(np.tan(theta)) + log_cos_lat

    sumcmpts = np.exp(logsumcmpts)


    grad_model = np.zeros((len(params), len(pi_mu)))
    params_i = 0
    reweight = (weights * np.exp(logcmpts)).T/sumcmpts
    for j in range(ncomponents):
        for par in fid_pars['free_pars'][j]:
            if par=='hz': grad_model[params_i] = grad_lambda[0,:,j].copy()*reweight[j]
            if par=='alpha3': grad_model[params_i] = grad_lambda[1,:,j].copy()*reweight[j]
            if par=='fD': grad_model[params_i] = grad_lambda[2,:,j].copy()*reweight[j]
            if par=='Mto': grad_model[params_i] = grad_lambda[3,:,j].copy()*reweight[j]
            if par=='w': grad_model[params_i] = np.exp(logcmpts[:,j])/sumcmpts
            params_i+=1
    for par in fid_pars['free_pars']['shd']:
        # if par=='alpha1': grad_model[params_i] = np.sum(grad_lambda[4,:,:].copy() * weights * np.exp(logcmpts), axis=1)/sumcmpts
        # if par=='alpha2': grad_model[params_i] = np.sum(grad_lambda[5,:,:].copy() * weights * np.exp(logcmpts), axis=1)/sumcmpts
        if par=='alpha1': grad_model[params_i] = np.sum(grad_lambda[4,:,:].copy() * reweight.T, axis=1)
        if par=='alpha2': grad_model[params_i] = np.sum(grad_lambda[5,:,:].copy() * reweight.T, axis=1)
        params_i+=1

    #print((grad_lambda[4,:,:] * weights * np.exp(logcmpts)).shape)

    jacobian = jacobian_params(params, fid_pars, ncomponents=ncomponents)

    return logsumcmpts + 2*np.log(np.tan(theta)) + log_cos_lat, (grad_model.T*jacobian).T

def log_expmodel_perr_grad(pi_mu, pi_err, abs_sin_lat, m_mu, log_pi_err, hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, smax=np.inf, degree=20, grad=False, integral_test=False, nu_norm=None):

    ep1=1.3; ep2=2.3;
    a1=-ln10*(ep1-1)/(2.5*alpha1); a2=-ln10*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    beta = abs_sin_lat/hz

    # Latent variables
    n1 = -(4 + alpha1*5/ln10)
    ng = -(4 + alphag*5/ln10)
    n2 = -(4 + alpha2*5/ln10)
    n3 = -(4 + alpha3*5/ln10)

    Ams_exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    Ams_coeff = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    log_Ams = np.log( fD*a1*a2 ) - scipy.special.logsumexp(Ams_exponent,b=Ams_coeff)
    log_AG = np.log(-alpha3) + np.log(1-fD)

    log_pnorm = -3*np.log(hz)

    # Absolute magnitude not known
    Mag_bounds = [-np.inf, Mto, Mms2, Mms1, Mx]
    Mag_n =      [n3,  n2,   ng,   n1]
    Mag_norm =   [log_AG  + alpha3*(Mto+10-m_mu),
                  log_Ams - np.log(a2) + alpha2*(Mms+10-m_mu),
                  log_Ams + np.log(Ag) + alphag*(Mms+10-m_mu),
                  log_Ams - np.log(a1) + alpha1*(Mms+10-m_mu)]

    if integral_test:
        return [beta, Mag_n, pi_mu, pi_err], Mag_norm, Mag_bounds

    p_model = np.zeros((4, len(pi_mu)))
    if grad:
        dp_model_dhz = np.zeros((4, len(pi_mu)))
        dp_model_dn = np.zeros((4, len(pi_mu)))
    for ii in range(4):

        a = np.exp((Mag_bounds[ii  ]+10-m_mu)*ln10/5)
        b = np.exp((Mag_bounds[ii+1]+10-m_mu)*ln10/5)
        n = Mag_n[ii]

        # Gauss - Hermite Quadrature
        args = (beta, n*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
        p_mode = functions.get_fooroots_ridder_hm(expmodel_perr_logit_grad, a=a+1e-15, b=b, args=np.array(args))
        curve = expmodel_perr_d2logIJ_dp2(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                    functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2
        z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
        sigma = 1/np.sqrt(-curve)
        p_integral = functions.integrate_gh_gap(expmodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=degree)
        p_model[ii] = p_integral.copy()

        if grad:
            # Gauss - Hermite Quadrature
            args = (beta, (n-1)*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
            p_mode = functions.get_fooroots_ridder_hm(expmodel_perr_logit_grad, a=a+1e-15, b=b, args=np.array(args))
            curve = expmodel_perr_d2logIJ_dp2(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                        functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2
            z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
            sigma = 1/np.sqrt(-curve)
            p_integral = functions.integrate_gh_gap(expmodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=degree)
            dp_model_dhz[ii] = p_integral
            # Gauss - Hermite Quadrature
            delta=1e-8
            args = (beta, (n+delta)*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
            p_mode = functions.get_fooroots_ridder_hm(expmodel_perr_logit_grad, a=a+1e-15, b=b, args=np.array(args))
            curve = expmodel_perr_d2logIJ_dp2(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                        functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2
            z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
            sigma = 1/np.sqrt(-curve)
            p_integral = functions.integrate_gh_gap(expmodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=degree)
            dp_model_dn[ii] = (p_integral-p_model[ii])/delta

    log_p = scipy.special.logsumexp(Mag_norm, b=p_model, axis=0)
    log_lambda = log_pnorm + log_p - 0.5*np.log(2*np.pi) - log_pi_err

    if not grad: return log_lambda

    exp_log_p = np.exp(log_p)

    grad_lambda = np.zeros((pi_mu.shape[0], 6)) + np.nan
    # hz
    grad_lambda[:,0] = -3/hz + abs_sin_lat/hz**2 * np.sum(dp_model_dhz*np.exp(Mag_norm), axis=0)/exp_log_p
    # alpha3
    grad_lambda[:,1] = np.exp(Mag_norm[0])*((1/alpha3 + Mto+10-m_mu)*p_model[0] - 5/ln10*dp_model_dn[0])/exp_log_p
    # fD
    grad_lambda[:,2] = ( p_model[0]*np.exp(Mag_norm[0])/(fD-1) + np.sum(p_model[1:]*np.exp(Mag_norm[1:]), axis=0)/fD )/exp_log_p

    # alpha 1/2
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
    Ams = np.exp(log_Ams)
    # alpha1
    dalphag_dalpha1 = (-1/alpha1 - (Mms-Mms1))/(Mms1-Mms2)
    dlnAms_dalpha1 = -1/alpha1 - np.sum(b_alpha1*np.exp(Ams_exponent))*Ams/(fD*a1*a2)
    grad_lambda[:,4] = (dlnAms_dalpha1 * np.sum(p_model[1:]*np.exp(Mag_norm[1:]), axis=0) \
                        + (np.exp(Mag_norm[3]) * ((1/alpha1 + (Mms+10-m_mu)) * p_model[3] \
                                                - 5/ln10 * dp_model_dn[3]) \
                        +  np.exp(Mag_norm[2]) * ((1/alpha1 + (Mms-Mms1)+dalphag_dalpha1*(Mms1+10-m_mu)) * p_model[2] \
                                                - 5/ln10 * dalphag_dalpha1 * dp_model_dn[2])))/exp_log_p
    # alpha2
    dalphag_dalpha2 = (1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)
    dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(Ams_exponent))*Ams/(fD*a1*a2)
    grad_lambda[:,5] = (dlnAms_dalpha2 * np.sum(p_model[1:]*np.exp(Mag_norm[1:]), axis=0) \
                        + (np.exp(Mag_norm[1]) * ((1/alpha2 + (Mms+10-m_mu)) * p_model[1] \
                                                - 5/ln10 * dp_model_dn[1]) \
                        +  np.exp(Mag_norm[2]) * (dalphag_dalpha2*(Mms1+10-m_mu) * p_model[2] \
                                                    - 5/ln10 * dalphag_dalpha2 * dp_model_dn[2])))/exp_log_p

    return log_lambda, grad_lambda.T

def expmodel_integrand(p, beta, n, h, mu, err):
    return p**n * np.exp(-beta/p)
def expmodel_perr_integrand(p, beta, n, mu, err):
    return p**n * np.exp(-beta/p - (p-mu)**2/(2*err**2))
@njit
def expmodel_perr_grad(p, args):
    beta, n, mu, err = args
    return p**3 - mu*p**2 - n*err**2*p - beta*err**2
@njit
def expmodel_perr_logit_grad(p, args):
    beta, n, mu, err, a, b = args
    return p**2 * (a+b-2*p) \
         + (n*p + beta - p**2*(p-mu)/err**2) * (p-a)*(b-p)
@njit
def expmodel_perr_d2logIJ_dp2(p, beta, n, mu, err, transform='none', b=None, a=None):
    d2logI_dp2 = -n/p**2 - 2*beta/p**3 - 1/err**2
    if   transform=='none':     return d2logI_dp2
    elif transform=='logit':    return d2logI_dp2 - 1/p**2 - 1/(p-b)**2
    elif transform=='logit_ab': return d2logI_dp2 - 1/(p-a)**2 - 1/(p-b)**2

def expmodel_perr_integrand_dn(p, beta, n, mu, err):
    return np.log(p) * p**n * np.exp(-beta/p - (p-mu)**2/(2*err**2))
@njit
def expmodel_perr_logit_grad_dn(p, args):
    beta, n, mu, err, a, b = args
    return p**2 * (a+b-2*p) + p*(p-a)*(b-p)/np.log(p) \
         + (n*p + beta - p**2*(p-mu)/err**2) * (p-a)*(b-p)
@njit
def expmodel_perr_d2logIJ_dp2_dn(p, beta, n, mu, err, transform='none', b=None, a=None):
    log_p = np.log(p)
    d2logI_dp2 = -(log_p + 1)/(p**2 * log_p**2) - n/p**2 - 2*beta/p**3 - 1/err**2
    if   transform=='none':     return d2logI_dp2
    elif transform=='logit':    return d2logI_dp2 - 1/p**2 - 1/(p-b)**2
    elif transform=='logit_ab': return d2logI_dp2 - 1/(p-a)**2 - 1/(p-b)**2

def log_halomodel_perr_grad(pi_mu, pi_err, abs_sin_lat, m_mu, log_pi_err, hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, smax=np.inf, degree=20, grad=False, integral_test=False, nu_norm=None):

    ep1=1.3; ep2=2.3;
    a1=-ln10*(ep1-1)/(2.5*alpha1); a2=-ln10*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    # Latent variables
    n1 = -(4 + alpha1*5/ln10)
    ng = -(4 + alphag*5/ln10)
    n2 = -(4 + alpha2*5/ln10)
    n3 = -(4 + alpha3*5/ln10)

    Ams_exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    Ams_coeff = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    log_Ams = np.log( fD*a1*a2 ) - scipy.special.logsumexp(Ams_exponent,b=Ams_coeff)
    log_AG = np.log(-alpha3) + np.log(1-fD)

    beta = abs_sin_lat/R0
    # log_pnorm = np.log(8) + scipy.special.gammaln(hz/2) - 3*np.log(R0) - 0.5*np.log(np.pi) - scipy.special.gammaln((hz-3)/2)
    log_pnorm = nu_norm(hz)

    # Absolute magnitude not known
    Mag_bounds = [-np.inf, Mto, Mms2, Mms1, Mx]
    Mag_n =      [n3,  n2,   ng,   n1]
    Mag_norm =   [log_AG  + alpha3*(Mto+10-m_mu),
                  log_Ams - np.log(a2) + alpha2*(Mms+10-m_mu),
                  log_Ams + np.log(Ag) + alphag*(Mms+10-m_mu),
                  log_Ams - np.log(a1) + alpha1*(Mms+10-m_mu)]

    if integral_test:
        return [beta, Mag_n, hz, pi_mu, pi_err], Mag_norm, Mag_bounds

    p_model = np.zeros((4, len(pi_mu)))
    if grad:
        dp_model_dhz = np.zeros((4, len(pi_mu)))
        dp_model_dn = np.zeros((4, len(pi_mu)))
    for ii in range(4):

        p_integral = np.zeros(len(pi_mu))

        n = Mag_n[ii]
        a = np.exp((Mag_bounds[ii  ]+10-m_mu)*ln10/5)
        a = np.where((1./smax)<a, a, 1./smax)
        b = np.exp((Mag_bounds[ii+1]+10-m_mu)*ln10/5)
        smax_subset = np.argwhere(a<b)[:,0]
        # print(f"{ii} subset {np.sum(1/smax<a)}, {len(smax_subset)}/{len(a)}")

        # Gauss - Hermite Quadrature
        args = (beta, n*np.ones(len(pi_mu)), hz*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
        args = [arg[smax_subset] for arg in args]
        p_mode = functions.get_fooroots_ridder_hm(halomodel_perr_logit_grad, a=args[-2]+1e-15, b=args[-1], args=np.array(args))
        curve = halomodel_perr_d2logIJ_dp2(p_mode, *args[:-2], transform='logit_ab', a=args[-2], b=args[-1]) / \
                                    functions.jac(p_mode, transform='logit_ab', a=args[-2], b=args[-1])**2

        z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
        sigma = 1/np.sqrt(-curve)
        p_integral = functions.integrate_gh_gap(halomodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=args[-2], b=args[-1], degree=degree)
        p_model[ii] = p_integral.copy()

        if grad:
            delta=1e-8
            # Gauss - Hermite Quadrature
            args = (beta, n*np.ones(len(pi_mu)), (hz+delta)*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
            args = [arg[smax_subset] for arg in args]
            p_mode = functions.get_fooroots_ridder_hm(halomodel_perr_logit_grad, a=args[-2]+1e-15, b=args[-1], args=np.array(args))
            curve = halomodel_perr_d2logIJ_dp2(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                        functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2

            z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
            sigma = 1/np.sqrt(-curve)
            p_integral = functions.integrate_gh_gap(halomodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=args[-2], b=args[-1], degree=degree)
            dp_model_dhz[ii] = (p_integral-p_model[ii])/delta
            # Gauss - Hermite Quadrature
            args = (beta, (n+delta)*np.ones(len(pi_mu)), hz*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
            args = [arg[smax_subset] for arg in args]
            p_mode = functions.get_fooroots_ridder_hm(halomodel_perr_logit_grad, a=args[-2]+1e-15, b=args[-1], args=np.array(args))
            curve = halomodel_perr_d2logIJ_dp2(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                        functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2

            z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
            sigma = 1/np.sqrt(-curve)
            p_integral = functions.integrate_gh_gap(halomodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=args[-2], b=args[-1], degree=degree)
            dp_model_dn[ii] = (p_integral-p_model[ii])/delta

    log_p = scipy.special.logsumexp(Mag_norm, b=p_model, axis=0)
    log_lambda = log_pnorm + log_p - 0.5*np.log(2*np.pi) - log_pi_err
    if not grad: return log_lambda

    exp_log_p = np.exp(log_p)

    grad_lambda = np.zeros((pi_mu.shape[0], 6)) + np.nan
    # hz
    # grad_lambda[:,0] = scipy.special.digamma(hz/2)/2 - scipy.special.digamma((hz-3)/2)/2 + np.sum(dp_model_dhz*np.exp(Mag_norm), axis=0)/exp_log_p
    dhz = hz*1e-5
    grad_lambda[:,0] = (nu_norm(hz+dhz)-nu_norm(hz))/dhz + np.sum(dp_model_dhz*np.exp(Mag_norm), axis=0)/exp_log_p
    # alpha3
    grad_lambda[:,1] = np.exp(Mag_norm[0])*((1/alpha3 + Mto+10-m_mu)*p_model[0] - 5/ln10*dp_model_dn[0])/exp_log_p
    # fD
    grad_lambda[:,2] = ( p_model[0]*np.exp(Mag_norm[0])/(fD-1) + np.sum(p_model[1:]*np.exp(Mag_norm[1:]), axis=0)/fD )/exp_log_p

    # alpha 1/2
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
    Ams = np.exp(log_Ams)
    # alpha1
    dalphag_dalpha1 = (-1/alpha1 - (Mms-Mms1))/(Mms1-Mms2)
    dlnAms_dalpha1 = -1/alpha1 - np.sum(b_alpha1*np.exp(Ams_exponent))*Ams/(fD*a1*a2)
    grad_lambda[:,4] = (dlnAms_dalpha1 * np.sum(p_model[1:]*np.exp(Mag_norm[1:]), axis=0) \
                        + (np.exp(Mag_norm[3]) * ((1/alpha1 + (Mms+10-m_mu)) * p_model[3] \
                                                - 5/ln10 * dp_model_dn[3]) \
                        +  np.exp(Mag_norm[2]) * ((1/alpha1 + (Mms-Mms1)+dalphag_dalpha1*(Mms1+10-m_mu)) * p_model[2] \
                                                - 5/ln10 * dalphag_dalpha1 * dp_model_dn[2])))/exp_log_p
    # alpha2
    dalphag_dalpha2 = (1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)
    dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(Ams_exponent))*Ams/(fD*a1*a2)
    grad_lambda[:,5] = (dlnAms_dalpha2 * np.sum(p_model[1:]*np.exp(Mag_norm[1:]), axis=0) \
                        + (np.exp(Mag_norm[1]) * ((1/alpha2 + (Mms+10-m_mu)) * p_model[1] \
                                                - 5/ln10 * dp_model_dn[1]) \
                        +  np.exp(Mag_norm[2]) * (dalphag_dalpha2*(Mms1+10-m_mu) * p_model[2] \
                                                    - 5/ln10 * dalphag_dalpha2 * dp_model_dn[2])))/exp_log_p

    return log_lambda, grad_lambda.T

@njit
def halomodel_perr_integrand(p, beta, n, h, mu, err):
    return p**n * (beta**2/p**2 + 1)**(-h/2) * np.exp(-(p-mu)**2/(2*err**2))
@njit
def halomodel_perr_grad(p, beta, n, h, mu, err):
    return -p**4 + mu*p**3 - (beta**2-n*err**2)*p**2 + mu*beta**2*p + (n+h)*err**2*beta**2
@njit
def halomodel_perr_logit_grad(p, args):
    beta, n, h, mu, err, a, b = args
    return p*(beta**2 + p**2) * (a+b-2*p) +\
          ((n - p*(p-mu)/err**2)*(beta**2+p**2) + h*beta**2) * (p-a)*(b-p)
@njit
def halomodel_perr_d2logIJ_dp2(p, beta, n, h, mu, err, transform='none', b=None, a=None):
    d2logI_dp2 = -(n+h)/p**2 + h*(p**2-beta**2)/(p**2+beta**2)**2 - 1/err**2
    if   transform=='none':     return d2logI_dp2
    elif transform=='logit':    return d2logI_dp2 - 1/p**2 - 1/(p-b)**2
    elif transform=='logit_ab': return d2logI_dp2 - 1/(p-a)**2 - 1/(p-b)**2

def halomodel_perr_integrand_dn(p, beta, n, h, mu, err):
    return np.log(p) * p**n * (beta**2/p**2 + 1)**(-h/2) * np.exp(-(p-mu)**2/(2*err**2))
@njit
def halomodel_perr_logit_grad_dn(p, args):
    beta, n, h, mu, err, a, b = args
    return p*(beta**2 + p**2) * (a+b-2*p) +\
          ((n + 1/np.log(p) - p*(p-mu)/err**2)*(beta**2+p**2) + h*beta**2) * (p-a)*(b-p)
@njit
def halomodel_perr_d2logIJ_dp2_dn(p, beta, n, h, mu, err, transform='none', b=None, a=None):
    log_p = np.log(p)
    d2logI_dp2 = -(log_p + 1)/(p**2 * log_p**2) -(n+h)/p**2 + h*(p**2-beta**2)/(p**2+beta**2)**2 - 1/err**2
    if   transform=='none':     return d2logI_dp2
    elif transform=='logit':    return d2logI_dp2 - 1/p**2 - 1/(p-b)**2
    elif transform=='logit_ab': return d2logI_dp2 - 1/(p-a)**2 - 1/(p-b)**2

def halomodel_perr_integrand_dh(p, beta, n, h, mu, err):
    return -0.5*np.log(beta**2/p**2 + 1) * p**n * (beta**2/p**2 + 1)**(-h/2) * np.exp(-(p-mu)**2/(2*err**2))
@njit
def halomodel_perr_logit_grad_dh(p, args):
    beta, n, h, mu, err, a, b = args
    return p*(beta**2 + p**2) * (a+b-2*p) +\
          ((n - p*(p-mu)/err**2)*(beta**2+p**2) - 2*beta**2/np.log(beta**2/p**2+1) + h*beta**2) * (p-a)*(b-p)

@njit
def halomodel_perr_d2logIJ_dp2_dh(p, beta, n, h, mu, err, transform='none', b=None, a=None):
    log_r = np.log(beta**2/p**2+1)
    d2logI_dp2 = (2*beta**2/log_r - (beta**2 + 3*p**2)) * 2*beta**2/( p**2*(beta**2+p**2)*log_r ) \
                 -(n+h)/p**2 + h*(p**2-beta**2)/(p**2+beta**2)**2 - 1/err**2
    if   transform=='none':     return d2logI_dp2
    elif transform=='logit':    return d2logI_dp2 - 1/p**2 - 1/(p-b)**2
    elif transform=='logit_ab': return d2logI_dp2 - 1/(p-a)**2 - 1/(p-b)**2

def halomodel_dist_trunc(smax, bmin, R0=8.27, directory=None):

    """
    Return function to evaluate halo normalisation as a function of n.
    """

    # For infinite smax, halo normalisation is analytic.
    if np.isinf(smax):
        #return lambda n: R0**3 * np.tan(np.deg2rad(bmin))**(-2) * (np.sqrt(np.pi)/8) * scipy.special.gamma((n_grid-3)/2)/scipy.special.gamma(n_grid/2)
        return lambda n: np.log(8) + scipy.special.gammaln(n/2) - 3*np.log(R0) - 0.5*np.log(np.pi) - scipy.special.gammaln((n-3)/2)

    file = f"smax{smax:.0f}_bmin{bmin:.0f}_R0{R0:.2f}.h"
    if os.path.exists(os.path.join(directory, file)):
        with h5py.File(os.path.join(directory, file), 'r') as hf:
            n = hf['n'][...]
            logI = hf['logI'][...]
    else:
        def halo_nu_integrand(sinb, s, n, R0=8.27):
            return s**2 * (sinb**2 * s**2/R0**2 + 1)**(-n/2)
        def halo_nu_integral(n_grid, smax, ss_scaled, sinbsinb, R0=8.27):
            # Rescale from 1:10 to 10^-2:smax
            I = np.zeros(n_grid.shape[0])
            ss = ss_scaled**(np.log10(smax)--2) / 100
            for i, n in tqdm.tqdm(enumerate(n_grid), total=len(n_grid)):
                integrand = halo_nu_integrand(sinbsinb, ss, n)
                I[i] = np.sum( (integrand[:-1,:-1] + integrand[1:,1:])/2 * (ss[0,1:]-ss[0,:-1])[None,:] * (sinbsinb[1:,0]-sinbsinb[:-1,0])[:,None] )
            return I
        n = np.linspace(1, 8, 1501)
        sinb = np.linspace(np.sin(np.deg2rad(bmin)),1,101)
        s_scaled = np.logspace(0,1,1001)
        ss_scaled, sinbsinb = np.meshgrid(s_scaled, sinb)
        logI = np.log(halo_nu_integral(n, smax, ss_scaled, sinbsinb)) + 2*np.log(np.tan(np.deg2rad(bmin)))
        # I = halo_nu_integral(n, smax, ss_scaled, sinbsinb)
        # print(I)
        # logI = np.log(I)

        with h5py.File(os.path.join(directory, file), 'w') as hf:
            hf.create_dataset('n', data=n)
            hf.create_dataset('logI', data=logI)

    return scipy.interpolate.interp1d(n, -logI)

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

def integral_model_cut(params, bins=None, fid_pars=None):

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    integralcmpts = np.zeros(ncomponents)
    if grad: integralcmpts_grad = np.zeros((ncomponents, 6))
    weights = np.zeros(ncomponents)
    integral_models = {'halo':integral_model_cut_halo,
                       'disk':integral_model_cut_disk}

    for j in range(ncomponents):
        _integral_model = integral_models[fid_pars['components'][j]]
        _uni_grid_integral = _integral_model(fid_pars['mmin'], fid_pars['mmax'], **transformed_params[j],
                                    Mx=fid_pars['Mmax'], R0=fid_pars['R0'], bmin=fid_pars['lat_min'], grad=grad)
        if grad: _uni_grid, _uni_grid_grad = _uni_grid_integral
        else: _uni_grid = _uni_grid_integral

        weights[j] = transformed_params[j]['w']

    if not grad: return np.sum(weights * _uni_grid)

def integral_model_cut_disk(mmin, mmax, hz=1., alpha1=-1., alpha2=-1., alpha3=-1., fD=0.5, Mto=4., Mms=8., Mms1=9., Mms2=7.,
                                        Mx=10., R0=8.27, bmin=np.pi/3, grad=False):

    betab = np.sin(bmin)/hz
    beta1 = 1/hz

    ep1=1.3; ep2=2.3;
    a1=-ln10*(ep1-1)/(2.5*alpha1); a2=-ln10*(ep2-1)/(2.5*alpha2);
    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    # Normalisation
    Ams_exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    Ams_coeff = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    Ams = a1*a2/np.sum(Ams_coeff*np.exp(Ams_exponent))
    AG = -alpha3

    # Absolute magnitude not known
    Mag_bounds = [-np.inf, Mto, Mms2, Mms1, Mx]
    Mag_alpha =  np.array([alpha3, alpha2, alphag, alpha1])
    Mag_t = 2 + 5*Mag_alpha/np.log(10)
    Mag_norm =   [(1-fD) * AG     * np.exp(alpha3*Mto),
                  fD     * Ams/a2 * np.exp(alpha2*Mms),
                  fD     * Ams*Ag * np.exp(alphag*Mms),
                  fD     * Ams/a1 * np.exp(alpha1*Mms)]

    I = np.zeros(4)
    for ii in range(4):
        smax0 = np.array([10**((mmax -(Mag_bounds[ii]  +10))/5)])
        smax1 = np.array([10**((mmax -(Mag_bounds[ii+1]  +10))/5)])
        smin0 = np.array([10**((mmin -(Mag_bounds[ii]+10))/5)])
        smin1 = np.array([10**((mmin -(Mag_bounds[ii+1]+10))/5)])

        I[ii] = (np.exp(-Mag_alpha[ii]*Mag_bounds[ii+1]) * ( \
                  + gamma_inc_rec_vecx(2, betab*smax1)/betab**2 \
                  - gamma_inc_rec_vecx(2, beta1*smax1)/beta1**2 ) \
             + np.exp(-Mag_alpha[ii]*(mmax-10)) * ( \
                  - (gamma_incc_rec_vecx(Mag_t[ii],betab*smax0) - gamma_incc_rec_vecx(Mag_t[ii],betab*smax1))/betab**(Mag_t[ii]) \
                  + (gamma_incc_rec_vecx(Mag_t[ii],beta1*smax0) - gamma_incc_rec_vecx(Mag_t[ii],beta1*smax1))/beta1**(Mag_t[ii]) ) \
             + np.exp(-Mag_alpha[ii]*Mag_bounds[ii]) * ( \
                  + gamma_incc_rec_vecx(2, betab*smax0)/betab**2 \
                  - gamma_incc_rec_vecx(2, beta1*smax0)/beta1**2 ) \
             - np.exp(-Mag_alpha[ii]*Mag_bounds[ii+1]) * ( \
                  + gamma_inc_rec_vecx(2,betab*smin1)/betab**(2) \
                  - gamma_inc_rec_vecx(2,beta1*smin1)/beta1**(2) ) \
             - np.exp(-Mag_alpha[ii]*(mmin-10)) * ( \
                  - (gamma_incc_rec_vecx(Mag_t[ii],betab*smin0) - gamma_incc_rec_vecx(Mag_t[ii],betab*smin1))/betab**Mag_t[ii] \
                  + (gamma_incc_rec_vecx(Mag_t[ii],beta1*smin0) - gamma_incc_rec_vecx(Mag_t[ii],beta1*smin1))/beta1**Mag_t[ii] ) \
             - np.exp(-Mag_alpha[ii]*Mag_bounds[ii]) * ( \
                  + gamma_incc_rec_vecx(2, betab*smin0)/betab**2 \
                  - gamma_incc_rec_vecx(2, beta1*smin0)/beta1**2 ) \
                  )[0] * (-hz/Mag_alpha[ii])

    return (np.tan(bmin)**2 / hz**3) * np.sum(I*Mag_norm)

def integral_model_cut_halo(params, bins=None, fid_pars=None):

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

def integral_model_gaiaSF_grad(params, bins=None, fid_pars=None, gsftest=None, test=False, grad=False):

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    gsf_pars = fid_pars['gsf_pars']
    _selectionfunction = np.zeros((len(gsf_pars['_m_grid']), len(gsf_pars['uni_n_pixels'])))
    for i in range(len(gsf_pars['uni_n_pixels'])):
        _selectionfunction[:,i] = sf_utils.gaia_sf(gsf_pars['_alpha'], gsf_pars['_beta'],
                                          gsf_pars['uni_n_pixels'][i], gsf_pars['_m_grid'])
    if gsftest is not None:
        gsf_pars = gsftest
        _selectionfunction = np.ones((len(gsf_pars['_m_grid']), len(gsf_pars['uni_n_pixels'])))

    if test: integral_pixel=np.zeros(len(gsf_pars['pixel_area']))

    integralcmpts = np.zeros(ncomponents)
    if grad: integralcmpts_grad = np.zeros((ncomponents, 6))
    weights = np.zeros(ncomponents)
    integral_models = {'halo':gaiasf_integrand_halo_grad,
                       'disk':gaiasf_integrand_disk_grad}
    for j in range(ncomponents):
        _integral_model = integral_models[fid_pars['components'][j]]
        _uni_grid_integral = _integral_model(gsf_pars['_m_grid'], gsf_pars['uni_sinb_pixels'], _selectionfunction,
                                    hz=transformed_params[j]['hz'],
                                    alpha1=transformed_params[j]['alpha1'], alpha2=transformed_params[j]['alpha2'], alpha3=transformed_params[j]['alpha3'],
                                    fD=transformed_params[j]['fD'], Mto=transformed_params[j]['Mto'], Mms=transformed_params[j]['Mms'],
                                    Mx=fid_pars['Mmax'], R0=fid_pars['R0'], smax=fit_pars['smax'], theta=fid_pars['lat_min'], grad=grad,
                                    nu_norm=fid_pars['halomodel_nu_norm'])

        if grad:
            _uni_grid, _uni_grid_grad = _uni_grid_integral
        else: _uni_grid = _uni_grid_integral

        # Summing over pixels is pretty fast!
        integralcmpts[j] = np.sum( _uni_grid[gsf_pars['idx_n_pixels'], gsf_pars['idx_sinb_pixels']]*gsf_pars['pixel_area']/(4*np.pi) )
        if test: integral_pixel += transformed_params[j]['w']*_uni_grid[gsf_pars['idx_n_pixels'], gsf_pars['idx_sinb_pixels']]*gsf_pars['pixel_area']/(4*np.pi)
        if grad: integralcmpts_grad[j] = transformed_params[j]['w']*np.sum( _uni_grid_grad[gsf_pars['idx_n_pixels'], gsf_pars['idx_sinb_pixels']].T*gsf_pars['pixel_area']/(4*np.pi), axis=1 )

        weights[j] = transformed_params[j]['w']

    if test: return integral_pixel
    if not grad: return np.sum(weights * integralcmpts)# * renorm

    integral_grad = np.zeros((len(params)))
    params_i = 0
    for j in range(ncomponents):
        for par in fid_pars['free_pars'][j]:
            if par=='hz': integral_grad[params_i] = integralcmpts_grad[j,0]
            if par=='alpha3': integral_grad[params_i] = integralcmpts_grad[j,1]
            if par=='fD': integral_grad[params_i] = integralcmpts_grad[j,2]
            if par=='Mto': integral_grad[params_i] = integralcmpts_grad[j,3]
            if par=='w': integral_grad[params_i] = integralcmpts[j]
            params_i+=1
    for par in fid_pars['free_pars']['shd']:
        if par=='alpha1': integral_grad[params_i] = np.sum(integralcmpts_grad[:,4])
        if par=='alpha2': integral_grad[params_i] = np.sum(integralcmpts_grad[:,5])
        params_i+=1

    jacobian = jacobian_params(params, fid_pars, ncomponents=ncomponents)

    return np.sum(weights * integralcmpts), integral_grad*jacobian

def gaiasf_integrand_disk_grad(m, sinb, _selectionfunction,
                          hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                          Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, smax=np.inf,
                          theta=np.pi/3, test=False, grad=False, nu_norm=None):

    # Overall normalisation
    norm = 2*np.tan(theta)**2

    ep1=1.3; ep2=2.3;
    a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    b = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    Ams = fD*a1*a2/np.sum(b*np.exp(exponent))
    AG = -alpha3*(1-fD)

    if grad:
        b_alpha1 = np.array([a2/alpha1 * ((Mms-Mms1)-1/alpha1),
                            -a2/alpha1 * ((Mms-Mx)  -1/alpha1),
                            a2/alphag * (-1/alpha1 + 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                           -a2/alphag * ((Mms-Mms1)+ 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                           -a1/(alpha1*alpha2),
                            a1/(alpha1*alpha2)])
        dlnAms_dalpha1 = -1/alpha1 - np.sum(b_alpha1*np.exp(exponent))*Ams/(fD*a1*a2)
        b_alpha2 = np.array([-a2/(alpha1*alpha2),
                             a2/(alpha1*alpha2),
                             a2/alphag * ((Mms-Mms2)- 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                             a2/alphag * ( 1/alpha2 + 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                             a1/alpha2 * ((Mms-Mto) - 1/alpha2),
                            -a1/alpha2 * ((Mms-Mms2)- 1/alpha2)])
        dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(exponent))*Ams/(fD*a1*a2)

    # Distance of magnitude break
    sx = np.exp((m - (Mx+10))*np.log(10)/5)
    sms1 = np.exp((m - (Mms1+10))*np.log(10)/5)
    sms2 = np.exp((m - (Mms2+10))*np.log(10)/5)
    sto = np.exp((m - (Mto+10))*np.log(10)/5)
    # Exponential disk profile distance powers
    t1 = 2+5*alpha1/np.log(10)
    tg = 2+5*alphag/np.log(10)
    t2 = 2+5*alpha2/np.log(10)
    t3 = 2+5*alpha3/np.log(10)
    # Exponential disk profile, apparent magnitude
    exp_a1 = np.exp((Mms+10-m)*alpha1)
    exp_ag = np.exp((Mms+10-m)*alphag)
    exp_a2 = np.exp((Mms+10-m)*alpha2)
    exp_a3 = np.exp((Mto+10-m)*alpha3)

    _uni_grid = np.zeros((_selectionfunction.shape[1], len(sinb)))
    if test: _uni_grid = np.zeros((len(m), len(sinb)))
    if grad: grad_I = np.zeros((len(sinb), 6, len(m)))

    for j in range(len(sinb)):
        #print('sinb: %d' % j)
        #print(t1, t2, t3)
        # Exponential disk profile
        disk_1 = gamma_incc_rec_vecx(t1+1, sx*sinb[j]/hz) - gamma_incc_rec_vecx(t1+1, sms1*sinb[j]/hz)
        disk_g = gamma_incc_rec_vecx(tg+1, sms1*sinb[j]/hz) - gamma_incc_rec_vecx(tg+1, sms2*sinb[j]/hz)
        disk_2 = gamma_incc_rec_vecx(t2+1, sms2*sinb[j]/hz) - gamma_incc_rec_vecx(t2+1, sto*sinb[j]/hz)
        disk_3 = gamma_incc_rec_vecx(t3+1, sto*sinb[j]/hz)
        #disk_3 = disk_integration_gaussquadlag(sto*sinb[j]/hz, np.zeros(len(sto))+10, t3)
        norm_disk_1 = Ams / a1 * (hz/sinb[j])**(t1+1) * exp_a1 / (2*hz**3)
        norm_disk_g = Ams * Ag * (hz/sinb[j])**(tg+1) * exp_ag / (2*hz**3)
        norm_disk_2 = Ams / a2 * (hz/sinb[j])**(t2+1) * exp_a2 / (2*hz**3)
        norm_disk_3 = AG  * (hz/sinb[j])**(t3+1) * exp_a3 / (2*hz**3)

        _integrand = norm* ( norm_disk_1*disk_1 + norm_disk_g*disk_g + norm_disk_2*disk_2 + norm_disk_3*disk_3)
        for i in range(_uni_grid.shape[0]):
            # Trapezium rule
            _uni_grid[i,j] = np.sum(0.5 * (_integrand[:-1]*_selectionfunction[:-1,i]\
                                        +  _integrand[1:] *_selectionfunction[1:,i])*(m[1:]-m[:-1]))
        if test: _uni_grid[:,j]=_integrand

        if grad:
            # hz
            ddisk1_dhz = ( (sx*sinb[j]/hz)**(t1+1)  *np.exp(-sx  *sinb[j]/hz)\
                         - (sms1*sinb[j]/hz)**(t1+1)*np.exp(-sms1*sinb[j]/hz) )/hz
            ddiskg_dhz = ( (sms1*sinb[j]/hz)**(tg+1)*np.exp(-sms1*sinb[j]/hz)\
                         - (sms2*sinb[j]/hz)**(tg+1)*np.exp(-sms2*sinb[j]/hz) )/hz
            ddisk2_dhz = ( (sms2*sinb[j]/hz)**(t2+1)*np.exp(-sms2*sinb[j]/hz)\
                         - (sto *sinb[j]/hz)**(t2+1)*np.exp(-sto *sinb[j]/hz) )/hz
            ddisk3_dhz = ( (sto *sinb[j]/hz)**(t3+1)*np.exp(-sto *sinb[j]/hz) )/hz
            grad_I[j,0,:] = norm*( ( disk_1*(t1-2)/hz + ddisk1_dhz )*norm_disk_1 \
                                      + ( disk_g*(tg-2)/hz + ddiskg_dhz )*norm_disk_g \
                                      + ( disk_2*(t2-2)/hz + ddisk2_dhz )*norm_disk_2 \
                                      + ( disk_3*(t3-2)/hz + ddisk3_dhz )*norm_disk_3 )
            # alpha3
            foo = lambda x: gamma_incc_rec_vecx(x+1, sto*sinb[j]/hz)
            #ddisk3_dt3 = (foo(t3+5e-11)-foo(t3-5e-11))/1e-10
            #ddisk3_dt3 = disk_integration_gaussquadlag(sto*sinb[j]/hz, np.zeros(len(sto))+10, t3, grad="t")
            ddisk3_dt3 = np.where(sto*sinb[j]/hz<1., (foo(t3+5e-11)-foo(t3-5e-11))/1e-10,
                                  disk_integration_gaussquadlag(sto*sinb[j]/hz, np.zeros(len(sto))+10, t3, grad="t") )
            grad_I[j,1,:] = norm*norm_disk_3*( (1/alpha3 + (Mto+10-m) + 5/np.log(10)*np.log(hz/sinb[j]))*disk_3\
                                                + 5/np.log(10) * ddisk3_dt3)
            # fD
            grad_I[j,2,:] = norm* ( (norm_disk_1*disk_1 + norm_disk_g*disk_g + norm_disk_2*disk_2)/fD + norm_disk_3*disk_3*alpha3/AG )
            # Mto
            grad_I[j,3,:] = norm*( (norm_disk_1*disk_1 + norm_disk_g*disk_g)*Ams*np.exp(alpha2*(Mms-Mto))/(fD*a2) \
                                  + norm_disk_2*( disk_2*Ams*np.exp(alpha2*(Mms-Mto))/(fD*a2) \
                                            - np.log(10)/5 * (sto*sinb[j]/hz)**(t2+1) * np.exp(-sto*sinb[j]/hz) ) \
                                  + norm_disk_3*( disk_3*alpha3 \
                                            + np.log(10)/5 * (sto*sinb[j]/hz)**(t3+1) * np.exp(-sto*sinb[j]/hz) ) )
            # alpha1, alpha2
            foo = lambda x: gamma_incc_rec_vecx(x+1, sx*sinb[j]/hz)-gamma_incc_rec_vecx(x+1, sms1*sinb[j]/hz)
            ddisk1_dalpha1 = 5/np.log(10) * (foo(t1+5e-11)-foo(t1+-5e-11))/1e-10#scipy.optimize.approx_fprime(t1,foo,1e-10)
            foo = lambda x: gamma_incc_rec_vecx(x+1, sms1*sinb[j]/hz)-gamma_incc_rec_vecx(x+1, sms2*sinb[j]/hz)
            ddiskg_dalphag = 5/np.log(10) * (foo(tg+5e-11)-foo(tg+-5e-11))/1e-10#scipy.optimize.approx_fprime(tg,foo,1e-10)
            foo = lambda x: gamma_incc_rec_vecx(x+1, sms2*sinb[j]/hz)-gamma_incc_rec_vecx(x+1, sto*sinb[j]/hz)
            ddisk2_dalpha2 = 5/np.log(10) * (foo(t2+5e-11)-foo(t2+-5e-11))/1e-10#scipy.optimize.approx_fprime(t2,foo,1e-10)

            dalphag_dalpha1 = (-1/alpha1 - (Mms-Mms1))/(Mms1-Mms2)
            dalphag_dalpha2 = (1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)
            # alpha1
            grad_I[j,4,:] = norm*( norm_disk_1*( ddisk1_dalpha1 \
                                              + disk_1*((dlnAms_dalpha1+1/alpha1) +   np.log(hz/sinb[j])*5/np.log(10) + (Mms+10-m)) ) \
                                    + norm_disk_g*( ddiskg_dalphag*dalphag_dalpha1 \
                                                   + disk_g*(dlnAms_dalpha1 + (1/alpha1 + (Mms-Mms1)*(1-dalphag_dalpha1))\
                                                            +(np.log(hz/sinb[j])*5/np.log(10) + (Mms+10-m))*dalphag_dalpha1) ) \
                                    + norm_disk_2 * dlnAms_dalpha1 * disk_2 )
            # alpha2
            grad_I[j,5,:] = norm*( norm_disk_1* dlnAms_dalpha2 * disk_1 \
                            + norm_disk_g*( ddiskg_dalphag*dalphag_dalpha2 \
                                           + disk_g*(dlnAms_dalpha2 + ((Mms-Mms1)*(-dalphag_dalpha2))\
                                                    +(np.log(hz/sinb[j])*5/np.log(10) + (Mms+10-m))*dalphag_dalpha2) ) \
                            + norm_disk_2 * ( ddisk2_dalpha2 \
                                              + disk_2*((dlnAms_dalpha2+1/alpha2) +   np.log(hz/sinb[j])*5/np.log(10) + (Mms+10-m))) )

    if not grad: return _uni_grid

    full_grad_I = np.zeros((_selectionfunction.shape[1], len(sinb), 6))
    for i in range(_uni_grid.shape[0]):
        # Trapezium rule
        full_grad_I[i] = np.sum(0.5 * (grad_I[:,:,:-1]*_selectionfunction[:-1,i]\
                                    +  grad_I[:,:,1:] *_selectionfunction[1:,i])*(m[1:]-m[:-1]), axis=2)

    return _uni_grid, full_grad_I #np.exp(log_lambda)*grad_lambda.T

def gaiasf_integrand_halo_grad(m, sinb, _selectionfunction,
                          hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                          Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, smax=np.inf,
                          theta=np.pi/3, test=False, grad=False, nu_norm=None):

    # Overall normalisation
    norm = 2*np.tan(theta)**2

    ep1=1.3; ep2=2.3;
    a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    b = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    Ams = fD*a1*a2/np.sum(b*np.exp(exponent))
    AG = -alpha3*(1-fD)

    if grad:
        b_alpha1 = np.array([a2/alpha1 * ((Mms-Mms1)-1/alpha1),
                            -a2/alpha1 * ((Mms-Mx)  -1/alpha1),
                            a2/alphag * (-1/alpha1 + 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                           -a2/alphag * ((Mms-Mms1)+ 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                           -a1/(alpha1*alpha2),
                            a1/(alpha1*alpha2)])
        dlnAms_dalpha1 = -1/alpha1 - np.sum(b_alpha1*np.exp(exponent))*Ams/(fD*a1*a2)
        b_alpha2 = np.array([-a2/(alpha1*alpha2),
                             a2/(alpha1*alpha2),
                             a2/alphag * ((Mms-Mms2)- 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                             a2/alphag * ( 1/alpha2 + 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                             a1/alpha2 * ((Mms-Mto) - 1/alpha2),
                            -a1/alpha2 * ((Mms-Mms2)- 1/alpha2)])
        dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(exponent))*Ams/(fD*a1*a2)

        dalphag_dalpha1 = (-1/alpha1 - (Mms-Mms1))/(Mms1-Mms2)
        dalphag_dalpha2 = ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)

    # Normalisation of halo component
    # gam_norm = 4*numba_special.vec_gamma(hz/2,hz/2)/(R0**3 * np.sqrt(np.pi) * numba_special.vec_gamma((hz-3)/2,(hz-3)/2))
    gam_norm = np.exp(nu_norm(hz))

    # Distance of magnitude break
    sx = np.exp((m - (Mx+10))*np.log(10)/5)
    sms1 = np.exp((m - (Mms1+10))*np.log(10)/5)
    sms2 = np.exp((m - (Mms2+10))*np.log(10)/5)
    sto = np.exp((m - (Mto+10))*np.log(10)/5)
    sx = np.where(sx<smax, sx, smax)
    sms1 = np.where(sms1<smax, sms1, smax)
    sms2 = np.where(sms2<smax, sms2, smax)
    sto = np.where(sto<smax, sto, smax)
    # Exponential disk profile distance powers
    t1 = 2+5*alpha1/np.log(10)
    tg = 2+5*alphag/np.log(10)
    t2 = 2+5*alpha2/np.log(10)
    t3 = 2+5*alpha3/np.log(10)
    # Exponential disk profile, apparent magnitude
    exp_a1 = np.exp((Mms+10-m)*alpha1) / a1
    exp_ag = np.exp((Mms+10-m)*alphag) * Ag
    exp_a2 = np.exp((Mms+10-m)*alpha2) / a2
    exp_a3 = np.exp((Mto+10-m)*alpha3)

    _uni_grid = np.zeros((_selectionfunction.shape[1], len(sinb)))
    #_uni_grid = np.zeros((len(m), len(sinb)))
    if grad: grad_I = np.zeros((len(sinb), 6, len(m)))

    for j in range(len(sinb)):
        # Power law halo profile
        # Integrate from sx to sms
        norm_halo_1 = Ams * gam_norm * exp_a1 * (R0/sinb[j])**(t1+1)
        halo_1 = halo_integration_gaussquad(sx *sinb[j]/R0, sms1*sinb[j]/R0, t1, hz)
        # Integrate from sx to sms
        norm_halo_g = Ams * gam_norm * exp_ag * (R0/sinb[j])**(tg+1)
        halo_g = halo_integration_gaussquad(sms1 *sinb[j]/R0, sms2*sinb[j]/R0, tg, hz)
        # Integrate from sms to sto
        norm_halo_2 = Ams * gam_norm * exp_a2 * (R0/sinb[j])**(t2+1)
        halo_2 = halo_integration_gaussquad(sms2*sinb[j]/R0, sto*sinb[j]/R0, t2, hz)
        # Integrate to inf
        norm_halo_3 = AG  * gam_norm * exp_a3 * (R0/sinb[j])**(t3+1)
        if alpha3>-1.: halo_3 = halo_integration_gaussquadlag(sto*sinb[j]/R0, np.inf,            t3, hz)
        else:          halo_3 = halo_integration_gaussquad(   sto*sinb[j]/R0, 10*sto*sinb[j]/R0, t3, hz)

        _integrand = norm*( norm_halo_1*halo_1 + norm_halo_g*halo_g + norm_halo_2*halo_2 + norm_halo_3*halo_3)

        for i in range(_uni_grid.shape[0]):
            # Trapezium rule
            _uni_grid[i,j] = np.sum(0.5 * (_integrand[:-1]*_selectionfunction[:-1,i]\
                                        +  _integrand[1:] *_selectionfunction[1:,i])*(m[1:]-m[:-1]))
        if test: _uni_grid[:,j]=_integrand
        #_uni_grid[:,j] = _integrand

        if grad:
            # hz
            # dlngamnorm_dhz = 0.5 * (scipy.special.digamma(hz/2) - scipy.special.digamma((hz-3)/2))
            dhz = hz*1e-5
            dlngamnorm_dhz = (nu_norm(hz+dhz)-nu_norm(hz))/dhz
            dhalo1_dhz = -0.5 * halo_integration_gaussquad(sx *sinb[j]/R0, sms1*sinb[j]/R0, t1, hz, grad="gamma")
            dhalog_dhz = -0.5*halo_integration_gaussquad(sms1 *sinb[j]/R0, sms2*sinb[j]/R0, tg, hz, grad="gamma")
            dhalo2_dhz = -0.5*halo_integration_gaussquad(sms2*sinb[j]/R0, sto*sinb[j]/R0, t2, hz, grad="gamma")
            if alpha3>-1.: dhalo3_dhz = -0.5*halo_integration_gaussquadlag(sto*sinb[j]/R0, np.inf,            t3, hz, grad="gamma")
            else:          dhalo3_dhz = -0.5*halo_integration_gaussquad(   sto*sinb[j]/R0, 10*sto*sinb[j]/R0, t3, hz, grad="gamma")
            grad_I[j,0,:] = norm*( ( halo_1*dlngamnorm_dhz + dhalo1_dhz )*norm_halo_1 \
                                 + ( halo_g*dlngamnorm_dhz + dhalog_dhz )*norm_halo_g \
                                 + ( halo_2*dlngamnorm_dhz + dhalo2_dhz )*norm_halo_2 \
                                 + ( halo_3*dlngamnorm_dhz + dhalo3_dhz )*norm_halo_3 )
            dhalo1_dalpha1 = 5/np.log(10) * halo_integration_gaussquad(sx   *sinb[j]/R0, sms1*sinb[j]/R0, t1, hz, grad="t")
            dhalog_dalphag = 5/np.log(10) * halo_integration_gaussquad(sms1 *sinb[j]/R0, sms2*sinb[j]/R0, tg, hz, grad="t")
            dhalo2_dalpha2 = 5/np.log(10) * halo_integration_gaussquad(sms2 *sinb[j]/R0, sto *sinb[j]/R0, t2, hz, grad="t")
            if alpha3>-1.:
                dhalo3_dalpha3 = 5/np.log(10)*halo_integration_gaussquadlag(sto*sinb[j]/R0, np.inf, t3, hz, grad="t")
            else:
                dhalo3_dalpha3 = 5/np.log(10)*halo_integration_gaussquad(sto*sinb[j]/R0, 10*sto*sinb[j]/R0, t3, hz, grad="t")
            # alpha3
            grad_I[j,1,:] = norm*norm_halo_3*( (1/alpha3 + (Mto+10-m) + 5/np.log(10)*np.log(R0/sinb[j]))*halo_3\
                                        + dhalo3_dalpha3 )
            # fD
            grad_I[j,2,:] = norm* ( (norm_halo_1*halo_1 + norm_halo_g*halo_g + norm_halo_2*halo_2)/fD + norm_halo_3*halo_3*alpha3/AG )
            # Mto
            grad_I[j,3,:] = norm*( (norm_halo_1*halo_1 + norm_halo_g*halo_g)*Ams*np.exp(alpha2*(Mms-Mto))/(fD*a2) \
                                  + norm_halo_2*( halo_2*Ams*np.exp(alpha2*(Mms-Mto))/(fD*a2) \
                                            - np.log(10)/5 * (sto*sinb[j]/R0)**(t2+1) * ((-sto*sinb[j]/R0)**2 + 1)**(-hz/2) ) \
                                  + norm_halo_3*( halo_3*alpha3 \
                                            + np.log(10)/5 * (sto*sinb[j]/R0)**(t3+1) *  ((-sto*sinb[j]/R0)**2 + 1)**(-hz/2) ) )
            # alpha1
            grad_I[j,4,:] = norm*( norm_halo_1*( dhalo1_dalpha1 \
                                              + halo_1*((dlnAms_dalpha1+1/alpha1) +   np.log(R0/sinb[j])*5/np.log(10) + (Mms+10-m)) ) \
                                    + norm_halo_g*( dhalog_dalphag*dalphag_dalpha1 \
                                                   + halo_g*(dlnAms_dalpha1 + (1/alpha1 + (Mms-Mms1)*(1-dalphag_dalpha1))\
                                                            +(np.log(R0/sinb[j])*5/np.log(10) + (Mms+10-m))*dalphag_dalpha1) ) \
                                    + norm_halo_2 * dlnAms_dalpha1 * halo_2 )
            # alpha2
            grad_I[j,5,:] = norm*( norm_halo_1* dlnAms_dalpha2 * halo_1 \
                            + norm_halo_g*( dhalog_dalphag*dalphag_dalpha2 \
                                           + halo_g*(dlnAms_dalpha2 + ((Mms-Mms1)*(-dalphag_dalpha2))\
                                                    +(np.log(R0/sinb[j])*5/np.log(10) + (Mms+10-m))*dalphag_dalpha2) ) \
                            + norm_halo_2 * ( dhalo2_dalpha2 \
                                              + halo_2*((dlnAms_dalpha2+1/alpha2) +   np.log(R0/sinb[j])*5/np.log(10) + (Mms+10-m))) )

    if not grad: return _uni_grid

    full_grad_I = np.zeros((_selectionfunction.shape[1], len(sinb), 6))
    for i in range(_uni_grid.shape[0]):
        # Trapezium rule
        full_grad_I[i] = np.sum(0.5 * (grad_I[:,:,:-1]*_selectionfunction[:-1,i]\
                                    +  grad_I[:,:,1:] *_selectionfunction[1:,i])*(m[1:]-m[:-1]), axis=2)

    return _uni_grid, full_grad_I

def appmag_model_subgaiaSF(params, fid_pars=None, sf=True):

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    gsf_pars = fid_pars['gsf_pars']

    integralcmpts = np.zeros((ncomponents, len(gsf_pars['_m_grid'])))
    weights = np.zeros(ncomponents)
    integral_models = {'halo':subgaiasf_integrand_halo_grad,
                       'disk':subgaiasf_integrand_disk_grad}
    for j in range(ncomponents):
        _integral_model = integral_models[fid_pars['components'][j]]
        integrand = _integral_model(gsf_pars['_m_grid'], gsf_pars['uni_sinb_pixels'],
                                    hz=transformed_params[j]['hz'],
                                    alpha1=transformed_params[j]['alpha1'], alpha2=transformed_params[j]['alpha2'], alpha3=transformed_params[j]['alpha3'],
                                    fD=transformed_params[j]['fD'], Mto=transformed_params[j]['Mto'], Mms=transformed_params[j]['Mms'],
                                    Mx=fid_pars['Mmax'], R0=fid_pars['R0'], theta=fid_pars['lat_min'], grad=False)

        # Summing over pixels is pretty fast!
        if sf: integrand = integrand[gsf_pars['idx_sinb_pixels']]*gsf_pars['_selectionfunction']
        else: integrand = integrand[gsf_pars['idx_sinb_pixels']]
        integralcmpts[j] = np.sum( integrand.T*gsf_pars['pixel_area']/(4*np.pi), axis=1 )

        weights[j] = transformed_params[j]['w']

    return (weights*integralcmpts.T).T

def integral_model_subgaiaSF_grad(params, bins=None, fid_pars=None, gsftest=None, test=False, grad=False):

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    gsf_pars = fid_pars['gsf_pars']

    if test: integral_pixel=np.zeros(len(gsf_pars['pixel_area']))

    integralcmpts = np.zeros(ncomponents)
    if grad: integralcmpts_grad = np.zeros((ncomponents, 6))
    weights = np.zeros(ncomponents)
    integral_models = {'halo':subgaiasf_integrand_halo_grad,
                       'disk':subgaiasf_integrand_disk_grad}
    for j in range(ncomponents):
        _integral_model = integral_models[fid_pars['components'][j]]
        integrand = _integral_model(gsf_pars['_m_grid'], gsf_pars['uni_sinb_pixels'],
                                    hz=transformed_params[j]['hz'],
                                    alpha1=transformed_params[j]['alpha1'], alpha2=transformed_params[j]['alpha2'], alpha3=transformed_params[j]['alpha3'],
                                    fD=transformed_params[j]['fD'], Mto=transformed_params[j]['Mto'], Mms=transformed_params[j]['Mms'],
                                    Mx=fid_pars['Mmax'], R0=fid_pars['R0'], smax=fid_pars['smax'], theta=fid_pars['lat_min'], grad=grad,
                                    nu_norm=fid_pars['halomodel_nu_norm'])

        if grad:
            integrand, integrand_grad = integrand
        else: integrand = integrand

        # Summing over pixels is pretty fast!
        integrand = integrand[gsf_pars['idx_sinb_pixels']]
        integrand = np.sum(0.5 * (integrand[:,:-1]*gsf_pars['_selectionfunction'][:,:-1]\
                               +  integrand[:,1:] *gsf_pars['_selectionfunction'][:,1:])*(gsf_pars['_m_grid'][1:]-gsf_pars['_m_grid'][:-1]),
                               axis=1)
        integralcmpts[j] = np.sum( integrand*gsf_pars['pixel_area']/(4*np.pi) )
        if grad:
            integrand_grad = np.moveaxis(integrand_grad[gsf_pars['idx_sinb_pixels']], 0,1)
            integrand_grad = np.sum(0.5 * (integrand_grad[:,:,:-1]*gsf_pars['_selectionfunction'][None,:,:-1]\
                                        +  integrand_grad[:,:,1:] *gsf_pars['_selectionfunction'][None,:,1:])*(gsf_pars['_m_grid'][1:]-gsf_pars['_m_grid'][None,:-1]),
                                       axis=2)
            integralcmpts_grad[j] = transformed_params[j]['w']*np.sum( integrand_grad*gsf_pars['pixel_area']/(4*np.pi), axis=1 )

        weights[j] = transformed_params[j]['w']

    if test: return integral_pixel
    if not grad: return np.sum(weights * integralcmpts)# * renorm

    integral_grad = np.zeros((len(params)))
    params_i = 0
    for j in range(ncomponents):
        for par in fid_pars['free_pars'][j]:
            if par=='hz': integral_grad[params_i] = integralcmpts_grad[j,0]
            if par=='alpha3': integral_grad[params_i] = integralcmpts_grad[j,1]
            if par=='fD': integral_grad[params_i] = integralcmpts_grad[j,2]
            if par=='Mto': integral_grad[params_i] = integralcmpts_grad[j,3]
            if par=='w': integral_grad[params_i] = integralcmpts[j]
            params_i+=1
    for par in fid_pars['free_pars']['shd']:
        if par=='alpha1': integral_grad[params_i] = np.sum(integralcmpts_grad[:,4])
        if par=='alpha2': integral_grad[params_i] = np.sum(integralcmpts_grad[:,5])
        params_i+=1

    jacobian = jacobian_params(params, fid_pars, ncomponents=ncomponents)

    return np.sum(weights * integralcmpts), integral_grad*jacobian

def subgaiasf_integrand_disk_grad(m, sinb,
                          hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                          Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, smax=np.inf,
                          theta=np.pi/3, test=False, grad=False, nu_norm=None):

    # Overall normalisation
    norm = 2*np.tan(theta)**2

    ep1=1.3; ep2=2.3;
    a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    b = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    Ams = fD*a1*a2/np.sum(b*np.exp(exponent))
    AG = -alpha3*(1-fD)

    if grad:
        b_alpha1 = np.array([a2/alpha1 * ((Mms-Mms1)-1/alpha1),
                            -a2/alpha1 * ((Mms-Mx)  -1/alpha1),
                            a2/alphag * (-1/alpha1 + 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                           -a2/alphag * ((Mms-Mms1)+ 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                           -a1/(alpha1*alpha2),
                            a1/(alpha1*alpha2)])
        dlnAms_dalpha1 = -1/alpha1 - np.sum(b_alpha1*np.exp(exponent))*Ams/(fD*a1*a2)
        b_alpha2 = np.array([-a2/(alpha1*alpha2),
                             a2/(alpha1*alpha2),
                             a2/alphag * ((Mms-Mms2)- 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                             a2/alphag * ( 1/alpha2 + 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                             a1/alpha2 * ((Mms-Mto) - 1/alpha2),
                            -a1/alpha2 * ((Mms-Mms2)- 1/alpha2)])
        dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(exponent))*Ams/(fD*a1*a2)

    # Distance of magnitude break
    sx = np.exp((m - (Mx+10))*np.log(10)/5)
    sms1 = np.exp((m - (Mms1+10))*np.log(10)/5)
    sms2 = np.exp((m - (Mms2+10))*np.log(10)/5)
    sto = np.exp((m - (Mto+10))*np.log(10)/5)
    # Exponential disk profile distance powers
    t1 = 2+5*alpha1/np.log(10)
    tg = 2+5*alphag/np.log(10)
    t2 = 2+5*alpha2/np.log(10)
    t3 = 2+5*alpha3/np.log(10)
    # Exponential disk profile, apparent magnitude
    exp_a1 = np.exp((Mms+10-m)*alpha1)
    exp_ag = np.exp((Mms+10-m)*alphag)
    exp_a2 = np.exp((Mms+10-m)*alpha2)
    exp_a3 = np.exp((Mto+10-m)*alpha3)

    integrand = np.zeros((len(sinb), len(m)))
    if grad: grad_integrand = np.zeros((len(sinb), 6, len(m)))

    for j in range(len(sinb)):

        # Exponential disk profile
        disk_1 = gamma_incc_rec_vecx(t1+1, sx*sinb[j]/hz) - gamma_incc_rec_vecx(t1+1, sms1*sinb[j]/hz)
        disk_g = gamma_incc_rec_vecx(tg+1, sms1*sinb[j]/hz) - gamma_incc_rec_vecx(tg+1, sms2*sinb[j]/hz)
        disk_2 = gamma_incc_rec_vecx(t2+1, sms2*sinb[j]/hz) - gamma_incc_rec_vecx(t2+1, sto*sinb[j]/hz)
        disk_3 = gamma_incc_rec_vecx(t3+1, sto*sinb[j]/hz)
        #disk_3 = disk_integration_gaussquadlag(sto*sinb[j]/hz, np.zeros(len(sto))+10, t3)
        norm_disk_1 = Ams / a1 * (hz/sinb[j])**(t1+1) * exp_a1 / (2*hz**3)
        norm_disk_g = Ams * Ag * (hz/sinb[j])**(tg+1) * exp_ag / (2*hz**3)
        norm_disk_2 = Ams / a2 * (hz/sinb[j])**(t2+1) * exp_a2 / (2*hz**3)
        norm_disk_3 = AG  * (hz/sinb[j])**(t3+1) * exp_a3 / (2*hz**3)

        integrand[j] = norm* ( norm_disk_1*disk_1 + norm_disk_g*disk_g + norm_disk_2*disk_2 + norm_disk_3*disk_3)

        if grad:
            # hz
            ddisk1_dhz = ( (sx*sinb[j]/hz)**(t1+1)  *np.exp(-sx  *sinb[j]/hz)\
                         - (sms1*sinb[j]/hz)**(t1+1)*np.exp(-sms1*sinb[j]/hz) )/hz
            ddiskg_dhz = ( (sms1*sinb[j]/hz)**(tg+1)*np.exp(-sms1*sinb[j]/hz)\
                         - (sms2*sinb[j]/hz)**(tg+1)*np.exp(-sms2*sinb[j]/hz) )/hz
            ddisk2_dhz = ( (sms2*sinb[j]/hz)**(t2+1)*np.exp(-sms2*sinb[j]/hz)\
                         - (sto *sinb[j]/hz)**(t2+1)*np.exp(-sto *sinb[j]/hz) )/hz
            ddisk3_dhz = ( (sto *sinb[j]/hz)**(t3+1)*np.exp(-sto *sinb[j]/hz) )/hz
            grad_integrand[j,0,:] = norm*( ( disk_1*(t1-2)/hz + ddisk1_dhz )*norm_disk_1 \
                                      + ( disk_g*(tg-2)/hz + ddiskg_dhz )*norm_disk_g \
                                      + ( disk_2*(t2-2)/hz + ddisk2_dhz )*norm_disk_2 \
                                      + ( disk_3*(t3-2)/hz + ddisk3_dhz )*norm_disk_3 )
            # alpha3
            foo = lambda x: gamma_incc_rec_vecx(x+1, sto*sinb[j]/hz)
            #ddisk3_dt3 = (foo(t3+5e-11)-foo(t3-5e-11))/1e-10
            #ddisk3_dt3 = disk_integration_gaussquadlag(sto*sinb[j]/hz, np.zeros(len(sto))+10, t3, grad="t")
            ddisk3_dt3 = np.where(sto*sinb[j]/hz<1., (foo(t3+5e-11)-foo(t3-5e-11))/1e-10,
                                  disk_integration_gaussquadlag(sto*sinb[j]/hz, np.zeros(len(sto))+10, t3, grad="t") )
            grad_integrand[j,1,:] = norm*norm_disk_3*( (1/alpha3 + (Mto+10-m) + 5/np.log(10)*np.log(hz/sinb[j]))*disk_3\
                                                + 5/np.log(10) * ddisk3_dt3)
            # fD
            grad_integrand[j,2,:] = norm* ( (norm_disk_1*disk_1 + norm_disk_g*disk_g + norm_disk_2*disk_2)/fD + norm_disk_3*disk_3*alpha3/AG )
            # Mto
            grad_integrand[j,3,:] = norm*( (norm_disk_1*disk_1 + norm_disk_g*disk_g)*Ams*np.exp(alpha2*(Mms-Mto))/(fD*a2) \
                                  + norm_disk_2*( disk_2*Ams*np.exp(alpha2*(Mms-Mto))/(fD*a2) \
                                            - np.log(10)/5 * (sto*sinb[j]/hz)**(t2+1) * np.exp(-sto*sinb[j]/hz) ) \
                                  + norm_disk_3*( disk_3*alpha3 \
                                            + np.log(10)/5 * (sto*sinb[j]/hz)**(t3+1) * np.exp(-sto*sinb[j]/hz) ) )
            # alpha1, alpha2
            foo = lambda x: gamma_incc_rec_vecx(x+1, sx*sinb[j]/hz)-gamma_incc_rec_vecx(x+1, sms1*sinb[j]/hz)
            ddisk1_dalpha1 = 5/np.log(10) * (foo(t1+5e-11)-foo(t1+-5e-11))/1e-10#scipy.optimize.approx_fprime(t1,foo,1e-10)
            foo = lambda x: gamma_incc_rec_vecx(x+1, sms1*sinb[j]/hz)-gamma_incc_rec_vecx(x+1, sms2*sinb[j]/hz)
            ddiskg_dalphag = 5/np.log(10) * (foo(tg+5e-11)-foo(tg+-5e-11))/1e-10#scipy.optimize.approx_fprime(tg,foo,1e-10)
            foo = lambda x: gamma_incc_rec_vecx(x+1, sms2*sinb[j]/hz)-gamma_incc_rec_vecx(x+1, sto*sinb[j]/hz)
            ddisk2_dalpha2 = 5/np.log(10) * (foo(t2+5e-11)-foo(t2+-5e-11))/1e-10#scipy.optimize.approx_fprime(t2,foo,1e-10)

            dalphag_dalpha1 = (-1/alpha1 - (Mms-Mms1))/(Mms1-Mms2)
            dalphag_dalpha2 = (1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)
            # alpha1
            grad_integrand[j,4,:] = norm*( norm_disk_1*( ddisk1_dalpha1 \
                                              + disk_1*((dlnAms_dalpha1+1/alpha1) +   np.log(hz/sinb[j])*5/np.log(10) + (Mms+10-m)) ) \
                                    + norm_disk_g*( ddiskg_dalphag*dalphag_dalpha1 \
                                                   + disk_g*(dlnAms_dalpha1 + (1/alpha1 + (Mms-Mms1)*(1-dalphag_dalpha1))\
                                                            +(np.log(hz/sinb[j])*5/np.log(10) + (Mms+10-m))*dalphag_dalpha1) ) \
                                    + norm_disk_2 * dlnAms_dalpha1 * disk_2 )
            # alpha2
            grad_integrand[j,5,:] = norm*( norm_disk_1* dlnAms_dalpha2 * disk_1 \
                            + norm_disk_g*( ddiskg_dalphag*dalphag_dalpha2 \
                                           + disk_g*(dlnAms_dalpha2 + ((Mms-Mms1)*(-dalphag_dalpha2))\
                                                    +(np.log(hz/sinb[j])*5/np.log(10) + (Mms+10-m))*dalphag_dalpha2) ) \
                            + norm_disk_2 * ( ddisk2_dalpha2 \
                                              + disk_2*((dlnAms_dalpha2+1/alpha2) +   np.log(hz/sinb[j])*5/np.log(10) + (Mms+10-m))) )

    if not grad: return integrand
    return integrand, grad_integrand

def subgaiasf_integrand_halo_grad(m, sinb,
                          hz=1., alpha1=-1., alpha2=-1., alpha3=-1.,
                          Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, smax=np.inf,
                          theta=np.pi/3, test=False, grad=False, nu_norm=None):

    # Overall normalisation
    norm = 2*np.tan(theta)**2

    ep1=1.3; ep2=2.3;
    a1=-np.log(10)*(ep1-1)/(2.5*alpha1); a2=-np.log(10)*(ep2-1)/(2.5*alpha2);

    alphag = (np.log(a1/a2) - alpha1*(Mms-Mms1) + alpha2*(Mms-Mms2))/(Mms1-Mms2)
    Ag = 1/a1 * np.exp((alpha1-alphag)*(Mms-Mms1))

    exponent = np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                        alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                        alpha2*(Mms-Mto), alpha2*(Mms-Mms2)])
    b = np.array([a2/alpha1, -a2/alpha1, a2/alphag, -a2/alphag, a1/alpha2, -a1/alpha2])
    Ams = fD*a1*a2/np.sum(b*np.exp(exponent))
    AG = -alpha3*(1-fD)

    if grad:
        b_alpha1 = np.array([a2/alpha1 * ((Mms-Mms1)-1/alpha1),
                            -a2/alpha1 * ((Mms-Mx)  -1/alpha1),
                            a2/alphag * (-1/alpha1 + 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                           -a2/alphag * ((Mms-Mms1)+ 1/alphag * ( 1/alpha1 + (Mms-Mms1))/(Mms1-Mms2)),
                           -a1/(alpha1*alpha2),
                            a1/(alpha1*alpha2)])
        dlnAms_dalpha1 = -1/alpha1 - np.sum(b_alpha1*np.exp(exponent))*Ams/(fD*a1*a2)
        b_alpha2 = np.array([-a2/(alpha1*alpha2),
                             a2/(alpha1*alpha2),
                             a2/alphag * ((Mms-Mms2)- 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                             a2/alphag * ( 1/alpha2 + 1/alphag * ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)),
                             a1/alpha2 * ((Mms-Mto) - 1/alpha2),
                            -a1/alpha2 * ((Mms-Mms2)- 1/alpha2)])
        dlnAms_dalpha2 = -1/alpha2 - np.sum(b_alpha2*np.exp(exponent))*Ams/(fD*a1*a2)

        dalphag_dalpha1 = (-1/alpha1 - (Mms-Mms1))/(Mms1-Mms2)
        dalphag_dalpha2 = ( 1/alpha2 + (Mms-Mms2))/(Mms1-Mms2)

    # Normalisation of halo component
    # gam_norm = 4*numba_special.vec_gamma(hz/2,hz/2)/(R0**3 * np.sqrt(np.pi) * numba_special.vec_gamma((hz-3)/2,(hz-3)/2))
    gam_norm = np.exp(nu_norm(hz))/2

    # Distance of magnitude break
    sx = np.exp((m - (Mx+10))*np.log(10)/5)
    sms1 = np.exp((m - (Mms1+10))*np.log(10)/5)
    sms2 = np.exp((m - (Mms2+10))*np.log(10)/5)
    sto = np.exp((m - (Mto+10))*np.log(10)/5)
    sx = np.where(sx<smax, sx, smax)
    sms1 = np.where(sms1<smax, sms1, smax)
    sms2 = np.where(sms2<smax, sms2, smax)
    sto = np.where(sto<smax, sto, smax)
    # Exponential disk profile distance powers
    t1 = 2+5*alpha1/np.log(10)
    tg = 2+5*alphag/np.log(10)
    t2 = 2+5*alpha2/np.log(10)
    t3 = 2+5*alpha3/np.log(10)
    # Exponential disk profile, apparent magnitude
    exp_a1 = np.exp((Mms+10-m)*alpha1) / a1
    exp_ag = np.exp((Mms+10-m)*alphag) * Ag
    exp_a2 = np.exp((Mms+10-m)*alpha2) / a2
    exp_a3 = np.exp((Mto+10-m)*alpha3)

    integrand = np.zeros((len(sinb), len(m)))
    if grad: grad_integrand = np.zeros((len(sinb), 6, len(m)))

    for j in range(len(sinb)):
        # Power law halo profile
        # Integrate from sx to sms
        norm_halo_1 = Ams * gam_norm * exp_a1 * (R0/sinb[j])**(t1+1)
        halo_1 = halo_integration_gaussquad(sx *sinb[j]/R0, sms1*sinb[j]/R0, t1, hz)
        # Integrate from sx to sms
        norm_halo_g = Ams * gam_norm * exp_ag * (R0/sinb[j])**(tg+1)
        halo_g = halo_integration_gaussquad(sms1 *sinb[j]/R0, sms2*sinb[j]/R0, tg, hz)
        # Integrate from sms to sto
        norm_halo_2 = Ams * gam_norm * exp_a2 * (R0/sinb[j])**(t2+1)
        halo_2 = halo_integration_gaussquad(sms2*sinb[j]/R0, sto*sinb[j]/R0, t2, hz)
        # Integrate to inf
        norm_halo_3 = AG  * gam_norm * exp_a3 * (R0/sinb[j])**(t3+1)
        if alpha3>-1.: halo_3 = halo_integration_gaussquadlag(sto*sinb[j]/R0, np.inf,            t3, hz)
        else:          halo_3 = halo_integration_gaussquad(   sto*sinb[j]/R0, 10*sto*sinb[j]/R0, t3, hz)

        integrand[j] = norm*( norm_halo_1*halo_1 + norm_halo_g*halo_g + norm_halo_2*halo_2 + norm_halo_3*halo_3)

        if grad:
            # hz
            # dlngamnorm_dhz = 0.5 * (scipy.special.digamma(hz/2) - scipy.special.digamma((hz-3)/2))
            dhz = hz*1e-5
            dlngamnorm_dhz = (nu_norm(hz+dhz)-nu_norm(hz))/dhz
            dhalo1_dhz = -0.5 * halo_integration_gaussquad(sx *sinb[j]/R0, sms1*sinb[j]/R0, t1, hz, grad="gamma")
            dhalog_dhz = -0.5*halo_integration_gaussquad(sms1 *sinb[j]/R0, sms2*sinb[j]/R0, tg, hz, grad="gamma")
            dhalo2_dhz = -0.5*halo_integration_gaussquad(sms2*sinb[j]/R0, sto*sinb[j]/R0, t2, hz, grad="gamma")
            if alpha3>-1.: dhalo3_dhz = -0.5*halo_integration_gaussquadlag(sto*sinb[j]/R0, np.inf,            t3, hz, grad="gamma")
            else:          dhalo3_dhz = -0.5*halo_integration_gaussquad(   sto*sinb[j]/R0, 10*sto*sinb[j]/R0, t3, hz, grad="gamma")
            grad_integrand[j,0,:] = norm*( ( halo_1*dlngamnorm_dhz + dhalo1_dhz )*norm_halo_1 \
                                 + ( halo_g*dlngamnorm_dhz + dhalog_dhz )*norm_halo_g \
                                 + ( halo_2*dlngamnorm_dhz + dhalo2_dhz )*norm_halo_2 \
                                 + ( halo_3*dlngamnorm_dhz + dhalo3_dhz )*norm_halo_3 )
            dhalo1_dalpha1 = 5/np.log(10) * halo_integration_gaussquad(sx   *sinb[j]/R0, sms1*sinb[j]/R0, t1, hz, grad="t")
            dhalog_dalphag = 5/np.log(10) * halo_integration_gaussquad(sms1 *sinb[j]/R0, sms2*sinb[j]/R0, tg, hz, grad="t")
            dhalo2_dalpha2 = 5/np.log(10) * halo_integration_gaussquad(sms2 *sinb[j]/R0, sto *sinb[j]/R0, t2, hz, grad="t")
            if alpha3>-1.:
                dhalo3_dalpha3 = 5/np.log(10)*halo_integration_gaussquadlag(sto*sinb[j]/R0, np.inf, t3, hz, grad="t")
            else:
                dhalo3_dalpha3 = 5/np.log(10)*halo_integration_gaussquad(sto*sinb[j]/R0, 10*sto*sinb[j]/R0, t3, hz, grad="t")
            # alpha3
            grad_integrand[j,1,:] = norm*norm_halo_3*( (1/alpha3 + (Mto+10-m) + 5/np.log(10)*np.log(R0/sinb[j]))*halo_3\
                                        + dhalo3_dalpha3 )
            # fD
            grad_integrand[j,2,:] = norm* ( (norm_halo_1*halo_1 + norm_halo_g*halo_g + norm_halo_2*halo_2)/fD + norm_halo_3*halo_3*alpha3/AG )
            # Mto
            grad_integrand[j,3,:] = norm*( (norm_halo_1*halo_1 + norm_halo_g*halo_g)*Ams*np.exp(alpha2*(Mms-Mto))/(fD*a2) \
                                  + norm_halo_2*( halo_2*Ams*np.exp(alpha2*(Mms-Mto))/(fD*a2) \
                                            - np.log(10)/5 * (sto*sinb[j]/R0)**(t2+1) * ((-sto*sinb[j]/R0)**2 + 1)**(-hz/2) ) \
                                  + norm_halo_3*( halo_3*alpha3 \
                                            + np.log(10)/5 * (sto*sinb[j]/R0)**(t3+1) *  ((-sto*sinb[j]/R0)**2 + 1)**(-hz/2) ) )
            # alpha1
            grad_integrand[j,4,:] = norm*( norm_halo_1*( dhalo1_dalpha1 \
                                              + halo_1*((dlnAms_dalpha1+1/alpha1) +   np.log(R0/sinb[j])*5/np.log(10) + (Mms+10-m)) ) \
                                    + norm_halo_g*( dhalog_dalphag*dalphag_dalpha1 \
                                                   + halo_g*(dlnAms_dalpha1 + (1/alpha1 + (Mms-Mms1)*(1-dalphag_dalpha1))\
                                                            +(np.log(R0/sinb[j])*5/np.log(10) + (Mms+10-m))*dalphag_dalpha1) ) \
                                    + norm_halo_2 * dlnAms_dalpha1 * halo_2 )
            # alpha2
            grad_integrand[j,5,:] = norm*( norm_halo_1* dlnAms_dalpha2 * halo_1 \
                            + norm_halo_g*( dhalog_dalphag*dalphag_dalpha2 \
                                           + halo_g*(dlnAms_dalpha2 + ((Mms-Mms1)*(-dalphag_dalpha2))\
                                                    +(np.log(R0/sinb[j])*5/np.log(10) + (Mms+10-m))*dalphag_dalpha2) ) \
                            + norm_halo_2 * ( dhalo2_dalpha2 \
                                              + halo_2*((dlnAms_dalpha2+1/alpha2) +   np.log(R0/sinb[j])*5/np.log(10) + (Mms+10-m))) )

    if not grad: return integrand
    return integrand, grad_integrand


@njit
def gamma_inc_rec_vecx(a, x):
    # Recurrence relation
    # Evaluate upper incomplete Gamma function for a<0
    # Initialise parameters
    ans=np.zeros(len(x));
    s=a;
    norm=1.;
    # Loop
    while s<=0:
        norm = norm/s
        ans += norm * x**s * np.exp(-x)
        s+=1;

    # Final answer
    ans += norm*numba_special.vec_gamma(s,s)*numba_special.vec_gammainc(s,x)

    return ans

@njit
def gamma_incc_rec_vecx(a, x):
    # Recurrence relation
    # Evaluate upper incomplete Gamma function for a<0
    # Initialise parameters
    ans=np.zeros(len(x));
    s=a;
    norm=1.;
    # Loop
    while s<=0:
        norm = norm/s
        ans += -norm * x**s * np.exp(-x)
        s+=1;

    # Final answer
    ans += norm*numba_special.vec_gamma(s,s)*numba_special.vec_gammaincc(s,x)

    return ans

@njit
def plfunction(x, n=0., gam=0.):
    return x**n * (x**2 + 1)**(-gam/2)
@njit
def betafunction(x, n=0., hz=0.):
    return x**n * np.exp(-x)
@njit
def disk_integration_gaussquadlag(x0,x1,n,grad=""):
    # Gauss-Laguerre Quadrature on x0-inf range
    ans=np.zeros(len(x0))
    jacobian = 1.
    for ii in range(degree):
        nodes_t = nodes_lag[ii] + x0
        if grad=="":
            ans += weights_lag[ii]*betafunction(nodes_t, n=n)*np.exp(nodes_lag[ii])*jacobian
        elif grad=="t":
            ans += weights_lag[ii]*betafunction(nodes_t, n=n)*np.exp(nodes_lag[ii])*jacobian\
                                  *np.log(nodes_t)
    return ans
@njit
def disk_integration_gaussquad(x0,x1,n,grad=""):
    # Gaussian Quadrature on x0-x1 range
    ans=np.zeros(len(x0))
    jacobian = (x1-x0)/2
    for ii in range(degree):
        nodes_t = (nodes_leg[ii]+1)*((x1-x0)/2) + x0
        if grad=="":
            ans += weights_leg[ii] * betafunction(nodes_t, n=n) * jacobian
        elif grad=="t":
            ans += weights_leg[ii] * betafunction(nodes_t, n=n) * jacobian\
                                   * np.log(nodes_t)
    return ans
@njit
def halo_integration_gaussquadlag(x0,x1,n,gam,grad=""):
    # Gauss-Laguerre Quadrature on x0-inf range
    ans=np.zeros(len(x0))
    jacobian = 1.
    for ii in range(degree):
        nodes_t = nodes_lag[ii] + x0
        if grad=="":
            ans += weights_lag[ii]*plfunction(nodes_t, n=n, gam=gam)*np.exp(nodes_lag[ii])*jacobian
        elif grad=="gamma":
            ans += weights_lag[ii]*plfunction(nodes_t, n=n, gam=gam)*np.exp(nodes_lag[ii])*jacobian\
                                  *np.log(nodes_t**2 + 1)
        elif grad=="t":
            ans += weights_lag[ii]*plfunction(nodes_t, n=n, gam=gam)*np.exp(nodes_lag[ii])*jacobian\
                                  *np.log(nodes_t)
    return ans
@njit
def halo_integration_gaussquad(x0,x1,n,gam,grad=""):
    # Gaussian Quadrature on x0-x1 range
    ans=np.zeros(len(x0))
    jacobian = (x1-x0)/2
    for ii in range(degree):
        nodes_t = (nodes_leg[ii]+1)*((x1-x0)/2) + x0
        if grad=="":
            ans += weights_leg[ii] * plfunction(nodes_t, n=n, gam=gam) * jacobian
        elif grad=="gamma":
            ans += weights_leg[ii] * plfunction(nodes_t, n=n, gam=gam) * jacobian\
                                  *np.log(nodes_t**2 + 1)
        elif grad=="t":
            ans += weights_leg[ii] * plfunction(nodes_t, n=n, gam=gam) * jacobian\
                                   * np.log(nodes_t)
    return ans


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
        for par in fid_pars['fixed_pars'][j].keys():
            output_pars[j][par]=fid_pars['fixed_pars'][j][par]
        for par in fid_pars['free_pars'][j]:
            if transform: output_pars[j][par]=fid_pars['functions'][j][par](params[params_i]); params_i += 1;
            else: output_pars[j][par]=params[params_i]; params_i += 1;
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
            else: jacobian[params_i]=params[params_i]; params_i += 1;
    for par in fid_pars['free_pars']['shd']:
        if transform: jacobian[params_i]=fid_pars['jacobians']['shd'][par](fid_pars['functions']['shd'][par](params[params_i]))
        else: jacobian[params_i]=params[params_i]
        params_i += 1

    return jacobian

#%% Priors
def model_prior(params, fid_pars=None, grad=False, bounds=None, dropoff=1e10):

    free_pars = fid_pars['free_pars']

    trans_params = combined_params(params, fid_pars, ncomponents=fid_pars['ncomponents'], transform=True)
    total_weight = np.sum([trans_params[j]['w'] for j in range(fid_pars['ncomponents'])])

    logprior = 0.;
    logprior_grad = np.zeros(len(params));

    prior_functions = {'none': lambda x: 0,
                       'logistic': lambda x: -x - 2*np.log(1+np.exp(-x)),
                       'dirichlet': lambda x, a: (a-1)*np.log(trans_params[j]['w']/total_weight)}
    prior_gradients = {'none': lambda x: 0,
                       'logistic': lambda x: (1-np.exp(x))/(1+np.exp(x)),
                       'dirichlet': lambda x, a: (a-1)*( 1 - fid_pars['ncomponents']*np.exp(params[params_i])/total_weight )}
    logistic = lambda x: -x - 2*np.log(1+np.exp(-x))
    logistic_grad = lambda x: (1-np.exp(x))/(1+np.exp(x))

    params_i = 0
    for j in range(fid_pars['ncomponents']):
        for par in fid_pars['free_pars'][j]:
            prior_args = fid_pars['priors'][j][par]
            logprior += prior_functions[prior_args[0]](params[params_i], *prior_args[1:])
            if grad: logprior_grad[params_i] += prior_gradients[prior_args[0]](params[params_i], *prior_args[1:])
            params_i += 1;
    for par in fid_pars['free_pars']['shd']:
        prior_args = fid_pars['priors']['shd'][par]
        logprior += prior_functions[prior_args[0]](params[params_i], *prior_args[1:])
        if grad: logprior_grad[params_i] += prior_gradients[prior_args[0]](params[params_i], *prior_args[1:])
        params_i += 1;

    boundary_prior = exponent_tophat(params, bounds, grad=grad, dropoff=dropoff)
    # unbound = (params<=bounds[0])|(params>=bounds[1])
    # if np.sum(unbound)>0:
    #     if not grad: return -1e30
    #     else: return -1e30, np.zeros(len(params))

    if not grad: return logprior+boundary_prior
    elif grad: return logprior + boundary_prior[0], logprior_grad + boundary_prior[1]

def exponent_tophat(x, bounds, grad=False, dropoff=1e10):

    d_lower =  (x-bounds[0]) * dropoff
    d_upper = -(x-bounds[1]) * dropoff

    prior = np.where(x<bounds[0], d_lower,
            np.where(x>bounds[1], d_upper, 0.))

    if not grad: return np.sum(prior)

    prior_grad = np.where(x<bounds[0],  dropoff,
                 np.where(x>bounds[1], -dropoff, 0.))

    return np.sum(prior), prior_grad


#%% Model Distribution
def z_component_models(z, hz=1., R0=8.27, bmin=80, smax=np.inf, component=None, directory=None):
    if component=='disk':
        norm = 1/(2*hz**3)
        dist = np.exp(-z/hz)
    elif component=='halo':
        norm_function = halomodel_dist_trunc(smax, bmin, R0=R0, directory=directory)
        norm = np.exp(norm_function(hz))/2
        # norm = 4*scipy.special.gamma(hz/2)/(R0**3 * np.sqrt(np.pi) * scipy.special.gamma((hz-3)/2))
        dist = ((z**2)/(R0**2) + 1)**(-hz/2)
        dist[z>smax]=0.

    return norm*dist

def z_model(z, params, fid_pars=None, model='combined', directory='/data/asfe2/Projects/mwtrace_data/utils/'):

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    dist_cmpts = np.zeros((len(z), ncomponents))
    weights = np.zeros(ncomponents)
    for j in range(ncomponents):
        dist_cmpts[:,j] = z_component_models(z, hz=transformed_params[j]['hz'], R0=fid_pars['R0'], smax=fid_pars['smax'],
                                                component=fid_pars['components'][j], directory=directory)
        weights[j] = transformed_params[j]['w']

    if model=='combined': return  z**2 * np.sum(weights*dist_cmpts, axis=1)
    elif model=='all':    return (z**2 *       (weights*dist_cmpts).T).T

def M_model(M, params, fid_pars=None, model='combined'):

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    dist_cmpts = np.zeros((len(M), ncomponents))
    weights = np.zeros(ncomponents)
    for j in range(ncomponents):
        alpha1=transformed_params[j]['alpha1']; alpha2=transformed_params[j]['alpha2']; alpha3=transformed_params[j]['alpha3']
        Mms=transformed_params[j]['Mms']; Mms1=transformed_params[j]['Mms1']; Mms2=transformed_params[j]['Mms2']; Mto=transformed_params[j]['Mto'];
        fD=transformed_params[j]['fD']; Mx=fid_pars['Mmax']

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

        log_Ams = np.log( fD*a1*a2 ) - \
                  scipy.special.logsumexp(np.array([alpha1*(Mms-Mms1), alpha1*(Mms-Mx),
                                                    alpha1*(Mms-Mms1)+alphag*(Mms1-Mms2), alpha1*(Mms-Mms1),
                                                    alpha2*(Mms-Mto), alpha2*(Mms-Mms2)]),
                                        b=np.array([a2/alpha1, -a2/alpha1,
                                                    a2/alphag, -a2/alphag,
                                                    a1/alpha2, -a1/alpha2]))
        log_AG = np.log(-alpha3) + np.log(1-fD)

        log_m = np.where(pop1, log_Ams - np.log(a1) + alpha1*(Mms-M),
                np.where(popg, log_Ams + np.log(Ag) + alphag*(Mms-M),
                np.where(pop2, log_Ams - np.log(a2) + alpha2*(Mms-M),
                               log_AG  + alpha3*(Mto-M))))

        m_dist = np.exp(log_m)
        m_dist[M>fid_pars['Mmax']]=0.
        dist_cmpts[:,j] = m_dist

        weights[j] = transformed_params[j]['w']

    if model=='combined': return  np.sum(weights*dist_cmpts, axis=1)
    elif model=='all':    return        (weights*dist_cmpts)


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
        beta, n, h, mu, err, a, b = test_args
        grad = lambda x: dh_msto.halomodel_perr_logit_grad(x, test_args) \
                       * dh_msto.halomodel_perr_integrand(x, *test_args[:-2])/(x*(beta**2+x**2))
        model = lambda x: (x-test_args[-2])*(test_args[-1]-x)*dh_msto.halomodel_perr_integrand(x, *test_args[:-2])

        self.assertAlmostEqual( grad(0.01), scipy.optimize.approx_fprime(np.array([0.01]), model, 1e-12), 8)
        self.assertAlmostEqual( grad(0.99), scipy.optimize.approx_fprime(np.array([0.99]), model, 1e-12), 8)

        test_args = (0.11459087188687923, -1.9315154043531053, 62.460138858539736,
                     -0.6133993246056172,  1.002678948333817, 0.2, 0.5709469567273955)
        beta, n, h, mu, err, a, b = test_args
        grad = lambda x: dh_msto.halomodel_perr_logit_grad(x, test_args) \
                       * dh_msto.halomodel_perr_integrand(x, *test_args[:-2])/(x*(beta**2+x**2))
        model = lambda x: (x-test_args[-2])*(test_args[-1]-x)*dh_msto.halomodel_perr_integrand(x, *test_args[:-2])
        self.assertAlmostEqual( grad(0.57), scipy.optimize.approx_fprime(np.array([0.57]), model, 1e-12), 8)
        self.assertAlmostEqual( grad(0.01), scipy.optimize.approx_fprime(np.array([0.01]), model, 1e-12), 8)

        grad = lambda x: dh_msto.halomodel_perr_logit_grad_dn(x, test_args) \
                       * dh_msto.halomodel_perr_integrand_dn(x, *test_args[:-2])/(x*(beta**2+x**2))
        model = lambda x: (x-test_args[-2])*(test_args[-1]-x)*dh_msto.halomodel_perr_integrand_dn(x, *test_args[:-2])
        self.assertAlmostEqual( grad(0.57), scipy.optimize.approx_fprime(np.array([0.57]), model, 1e-12), 8)
        self.assertAlmostEqual( grad(0.01), scipy.optimize.approx_fprime(np.array([0.01]), model, 1e-12), 8)

        grad = lambda x: dh_msto.halomodel_perr_logit_grad_dh(x, test_args) \
                       * dh_msto.halomodel_perr_integrand_dh(x, *test_args[:-2])/(x*(beta**2+x**2))
        model = lambda x: (x-test_args[-2])*(test_args[-1]-x)*dh_msto.halomodel_perr_integrand_dh(x, *test_args[:-2])
        self.assertAlmostEqual( grad(0.57), scipy.optimize.approx_fprime(np.array([0.57]), model, 1e-12), 8)
        self.assertAlmostEqual( grad(0.01), scipy.optimize.approx_fprime(np.array([0.01]), model, 1e-12), 8)

        test_args = (1,3,2.,0.5,0.1,0.,0.,1.)
        model = lambda h: dh_msto.halomodel_perr_integrand(0.1, *(1,3,h,0.5,0.1))
        grad = lambda h: dh_msto.halomodel_perr_integrand_dh(0.1, *(1,3,h,0.5,0.1))
        self.assertAlmostEqual( grad(0.1), scipy.optimize.approx_fprime(np.array([0.1]), model, 1e-12), 8)

        test_args = (1,3,2.,0.5,0.1,0.,0.,1.)
        model = lambda n: dh_msto.halomodel_perr_integrand(0.1, *(1,n,2.,0.5,0.1))
        grad = lambda n: dh_msto.halomodel_perr_integrand_dn(0.1, *(1,n,2.,0.5,0.1))
        self.assertAlmostEqual( grad(0.1), scipy.optimize.approx_fprime(np.array([0.1]), model, 1e-12), 8)

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

    def test_priors(self):

        # transform, p1, p2, lower bound, upper bound
        param_trans = {}
        a_dirichlet = 2
        param_trans['shd'] = {'alpha1':('nexp',0,0,-3,3,'none'),
                              'alpha2':('nexp',0,0,-3,3,'none')}
        param_trans[0] = {'w':('exp',0,0,-10,10,'dirichlet',a_dirichlet),
                          'fD': ('logit_scaled', 0,1, -10,10,'logistic'),
                          'alpha3':('nexp',0,0,-10,10,'none'),
                          'hz': ('logit_scaled', 0,  1.2,-10,10,'logistic')}
        param_trans[1] = {'w':('exp',0,0,-10,10,'dirichlet',a_dirichlet),
                          'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                          'alpha3':('nexp',0,0,-10,10,'none'),
                          'hz': ('logit_scaled', 1.2,3,-10,10,'logistic')}
        param_trans[2] = {'w':('exp',0,0,-10,10,'dirichlet',a_dirichlet),
                          'fD': ('logit_scaled', 0,1,-10,10,'logistic'),
                          'alpha3':('nexp',0,0,-10,10,'none'),
                          'hz': ('logit_scaled', 3,  7.3,-10,10,'logistic')}

        fid_pars['free_pars'][0] = ['w', 'hz', 'fD']
        fid_pars['free_pars'][1] = ['w', 'hz', 'fD']
        fid_pars['free_pars'][2] = ['w', 'hz', 'fD']
        fid_pars['free_pars']['shd'] = ['alpha1', 'alpha2']
        ndim=np.sum([len(fid_pars['free_pars'][key]) for key in fid_pars['free_pars'].keys()])

        fid_pars['priors'] = {}
        params_i = 0
        for cmpt in np.arange(fid_pars['ncomponents']).tolist()+['shd',]:
            fid_pars['priors'][cmpt]={};
            for par in fid_pars['free_pars'][cmpt]:
                fid_pars['priors'][cmpt][par] = param_trans[cmpt][par][5:]
                params_i += 1;

        p0 = np.array( [transformations.logit(np.random.rand()),transformations.logit(np.random.rand()),-np.random.rand()*1,
                        transformations.logit(np.random.rand()),transformations.logit(np.random.rand()),-np.random.rand()*1,
                        transformations.logit(np.random.rand()),transformations.logit(np.random.rand()),-np.random.rand()*1,
                        -np.random.rand()*1,-np.random.rand()*1] )

        model = lambda x: dh_msto.model_prior(x, fid_pars=fid_pars, grad=False)
        grad = lambda x: dh_msto.model_prior(x, fid_pars=fid_pars, grad=True)[1]
        scipy.optimize.approx_fprime(p0, model, 1e-8), grad(p0)
        self.assertAlmostEqual( grad(p0), scipy.optimize.approx_fprime(np.array([p0]), model, 1e-12), 8)


if __name__ == '__main__':
    unittest.main()
