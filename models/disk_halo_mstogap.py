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

ln10 = np.log(10)


def logmodel_grad(sample, params, gmm=None, fid_pars=None, grad=False):

    # Observables
    pi_mu, log_pi_mu, abs_sin_lat, log_cos_lat, m_mu = sample

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    # Defined paramers
    theta=fid_pars['lat_min']; Mx=fid_pars['Mmax']; R0=fid_pars['R0']

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
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, grad=False):

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


def logmodel_perr_grad(sample, params, gmm=None, fid_pars=None, grad=False):

    # Observables
    pi_mu, pi_err, abs_sin_lat, log_cos_lat, m_mu, log_pi_err = sample

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    # Defined paramers
    theta=fid_pars['lat_min']; Mx=fid_pars['Mmax']; R0=fid_pars['R0']

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
                                                Mx=Mx, R0=R0, grad=grad)

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
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, degree=21, grad=False):

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

    p_model = np.zeros((4, len(pi_mu)))
    for ii in range(4):

        p_integral = np.zeros(len(pi_mu))

        p_min = np.exp((Mag_bounds[ii  ]+10-m_mu)*ln10/5)
        p_max = np.exp((Mag_bounds[ii+1]+10-m_mu)*ln10/5)

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
        p_integral[~legendre] = functions.integrate_gh_gap(expmodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=degree)

        p_model[ii] = p_integral

    log_p = scipy.special.logsumexp(Mag_norm, b=p_model, axis=0)
    log_lambda = log_pnorm + log_p - 0.5*np.log(2*np.pi) - log_pi_err

    if not grad: return log_lambda

    exp_log_p = np.exp(log_p)

    grad_lambda = np.zeros((pi_mu.shape[0], 6)) + np.nan
    # hz
    dp_model_dhz = np.zeros((4, len(pi_mu)))
    for ii in range(4): # Run integration with n -> n-1
        p_integral = np.zeros(len(pi_mu))
        a = np.exp((Mag_bounds[ii  ]+10-m_mu)*ln10/5)
        b = np.exp((Mag_bounds[ii+1]+10-m_mu)*ln10/5)
        n = Mag_n[ii]-1
        # Gauss - Hermite Quadrature
        args = (beta, n*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
        p_mode = functions.get_fooroots_ridder_hm(expmodel_perr_logit_grad, a=a+1e-15, b=b, args=args)
        curve = expmodel_perr_d2logIJ_dp2(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                    functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2
        z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
        sigma = 1/np.sqrt(-curve)
        p_integral = functions.integrate_gh_gap(expmodel_perr_integrand, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=degree)
        dp_model_dhz[ii] = p_integral
    grad_lambda[:,0] = -3/hz + abs_sin_lat/hz**2 * np.sum(dp_model_dhz*np.exp(Mag_norm), axis=0)/exp_log_p
    # n
    dp_model_dn = np.zeros((4, len(pi_mu)))
    for ii in range(4): # Run integration with log(p)p^n
        p_integral = np.zeros(len(pi_mu))
        a = np.exp((Mag_bounds[ii  ]+10-m_mu)*ln10/5)
        b = np.exp((Mag_bounds[ii+1]+10-m_mu)*ln10/5)
        n = Mag_n[ii]
        # Gauss - Hermite Quadrature
        args = (beta, n*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
        p_mode = functions.get_fooroots_ridder_hm(expmodel_perr_logit_grad_dn, a=a+1e-15, b=b, args=args)
        curve = expmodel_perr_d2logIJ_dp2_dn(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                    functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2
        z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
        sigma = 1/np.sqrt(-curve)
        p_integral = functions.integrate_gh_gap(expmodel_perr_integrand_dn, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=degree)
        dp_model_dn[ii] = p_integral
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
                                Mto=4., Mms=8., Mms1=9., Mms2=7., fD=0.5, Mx=10., R0=8.27, degree=21, grad=False):

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
        p_min = np.exp((Mag_bounds[ii  ]+10-m_mu)*ln10/5)
        p_max = np.exp((Mag_bounds[ii+1]+10-m_mu)*ln10/5)

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
    log_lambda = log_pnorm + log_p - 0.5*np.log(2*np.pi) - log_pi_err
    if not grad: return log_lambda

    exp_log_p = np.exp(log_p)

    grad_lambda = np.zeros((pi_mu.shape[0], 6)) + np.nan
    # hz
    dp_model_dhz = np.zeros((4, len(pi_mu)))
    for ii in range(4): # Run integration with n -> n-1
        p_integral = np.zeros(len(pi_mu))
        a = np.exp((Mag_bounds[ii  ]+10-m_mu)*ln10/5)
        b = np.exp((Mag_bounds[ii+1]+10-m_mu)*ln10/5)
        n = Mag_n[ii]
        # Gauss - Hermite Quadrature
        args = (beta, n*np.ones(len(pi_mu)), hz*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
        p_mode = functions.get_fooroots_ridder_hm(halomodel_perr_logit_grad_dh, a=a+1e-15, b=b, args=args)
        curve = halomodel_perr_d2logIJ_dp2_dh(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                    functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2
        z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
        sigma = 1/np.sqrt(-curve)
        p_integral = functions.integrate_gh_gap(halomodel_perr_integrand_dh, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=degree)
        dp_model_dhz[ii] = p_integral
    grad_lambda[:,0] = scipy.special.digamma(hz/2)/2 - scipy.special.digamma((hz-3)/2)/2 + np.sum(dp_model_dhz*np.exp(Mag_norm), axis=0)/exp_log_p
    # n
    dp_model_dn = np.zeros((4, len(pi_mu)))
    for ii in range(4): # Run integration with log(p)p^n
        p_integral = np.zeros(len(pi_mu))
        a = np.exp((Mag_bounds[ii  ]+10-m_mu)*ln10/5)
        b = np.exp((Mag_bounds[ii+1]+10-m_mu)*ln10/5)
        n = Mag_n[ii]
        # Gauss - Hermite Quadrature
        args = (beta, n*np.ones(len(pi_mu)), hz*np.ones(len(pi_mu)), pi_mu, pi_err, a, b)
        p_mode = functions.get_fooroots_ridder_hm(halomodel_perr_logit_grad_dn, a=a+1e-15, b=b, args=args)
        curve = halomodel_perr_d2logIJ_dp2_dn(p_mode, *args[:-2], transform='logit_ab', a=a, b=b) / \
                                    functions.jac(p_mode, transform='logit_ab', a=a, b=b)**2
        z_mode = functions.trans(p_mode, transform='logit_ab', a=a, b=b)
        sigma = 1/np.sqrt(-curve)
        p_integral = functions.integrate_gh_gap(halomodel_perr_integrand_dn, z_mode, sigma, args[:-2], transform='logit_ab', a=a, b=b, degree=degree)
        dp_model_dn[ii] = p_integral
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
def model_prior(params, fid_pars=None, grad=False, bounds=None):

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

    unbound = (params<=bounds[0])|(params>=bounds[1])
    if np.sum(unbound)>0:
        if not grad: return -1e30
        else: return -1e30, np.zeros(len(params))

    if not grad: return logprior
    elif grad: return logprior, logprior_grad


#%% Model Distribution
def z_component_models(z, hz=1., R0=8.27, component=None):
    if component=='disk':
        norm = 1/(2*hz**3)
        dist = np.exp(-z/hz)
    elif component=='halo':
        norm = 4*scipy.special.gamma(hz/2)/(R0**3 * np.sqrt(np.pi) * scipy.special.gamma((hz-3)/2))
        dist = ((z**2)/(R0**2) + 1)**(-hz/2)

    return norm*dist

def z_model(z, params, fid_pars=None, model='combined'):

    # Input Parameters
    ncomponents=fid_pars['ncomponents']
    transformed_params = combined_params(params, fid_pars, ncomponents=ncomponents)

    dist_cmpts = np.zeros((len(z), ncomponents))
    weights = np.zeros(ncomponents)
    for j in range(ncomponents):
        dist_cmpts[:,j] = z_component_models(z, hz=transformed_params[j]['hz'], R0=fid_pars['R0'], component=fid_pars['components'][j])
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
