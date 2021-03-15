import sys, os, numpy as np, tqdm
from numba import njit



#@njit
def get_roots(coeffs, b=None):
    roots = np.zeros(coeffs.shape[0]) + np.nan
    ngood = np.zeros(coeffs.shape[0]) + np.nan
    if b is None:
        b = np.zeros(coeffs.shape[0]) + np.inf
    for i in range(coeffs.shape[0]):
        rootsi = np.roots(coeffs[i,:])
        im = np.imag(rootsi)
        re = np.real(rootsi)
        good_root = (np.abs(im/re)<1e-10)&((re-b[i])/b[i]<-1e-12)&(re>0)
        root = re[good_root][0]
        roots[i]=root
        ngood[i]=np.sum(good_root)

    return roots, ngood

def integrand(p, beta, pi_mu, pi_err, n):
    return p**n * np.exp(-beta/p - ((p-pi_mu)**2/(2*pi_err**2)))

#@njit
def trans(p, transform='none', b=None, a=None):
    if transform=='none':    return p
    elif transform=='log':   return np.log(p)
    elif transform=='log_b': return np.log(p)
    elif transform=='logit': return np.log(p/(b-p))
    elif transform=='logit_ab': return np.log((p-a)/(b-p))
#@njit
def trans_i(z, transform='none', b=None, a=None):
    if transform=='none':    return  z
    elif transform=='log':   return np.exp(z)
    elif transform=='log_b': return np.exp(z)
    elif transform=='logit': return b/(1+np.exp(-z))
    elif transform=='logit_ab': return (a+b*np.exp(z))/(1+np.exp(z))
#@njit
def jac(p, transform='none', b=None, a=None):
    if transform=='none':    return p
    elif transform=='log':   return 1/p
    elif transform=='log_b': return 1/p
    elif transform=='logit': return b/(p*(b-p))
    elif transform=='logit_ab': return (b-a)/((p-a)*(b-p))
def dIJdp(p, beta, pi_mu, pi_err, n, transform='none', b=None, a=None):
    if transform=='none': return p**3 - pi_mu*p**2 - n*pi_err**2 * p - beta*pi_err**2
    elif transform=='log': return p**3 - pi_mu*p**2 - (n+1)*pi_err**2 * p - beta*pi_err**2
    elif transform=='logit':
        return p**4 - (b+pi_mu)*p**3 + (b*pi_mu-(n+2)*pi_err**2) * p**2 \
                                     + (b*(n+1)-beta)*pi_err**2  * p   + b*beta*pi_err**2

def dIJdp_fprime(p, beta, pi_mu, pi_err, n, transform='none', b=None):
    if transform=='none': return 3*p**2 - 2*pi_mu*p - n*pi_err**2
    elif transform=='log': return 3*p**2 - 2*pi_mu*p - (n+1)*pi_err**2
    elif transform=='logit':
        return 4*p**3 - 3*(b+pi_mu)*p**2 + 2*(b*pi_mu-(n+2)*pi_err**2) * p + (b*(n+1)-beta)*pi_err**2

@njit
def d2logIJ_dp2(p, beta, pi_mu, pi_err, n, transform='none', b=None):
    if   transform=='none':  return             - n/p**2     - 2*beta/p**3     - 1/pi_err**2
    elif transform=='log':   return             -(n+1)/p**2  - 2*beta/p**3     - 1/pi_err**2
    elif transform=='log_b': return -1/p**2     - n/(p+b)**2 - 2*beta/(p+b)**3 - 1/pi_err**2
    elif transform=='logit': return -1/(b-p)**2 - (n+1)/p**2 - 2*beta/p**3     - 1/pi_err**2


def integrate_gh(integrand, jacobian, inverse_transform, root_z, sigma, args, kwargs,
                 degree=10):

    """
    Gauss-Hermite integral
    """

    if (kwargs['transform']=='logit')|(kwargs['transform']=='log_b'):
        if np.isscalar(kwargs['b']): b = kwargs['b']
        else: b = kwargs['b'][:,np.newaxis]

    # Draw nodes and weights for Gauss-Hermite.
    nodes, weights = np.polynomial.hermite.hermgauss(degree)
    # Transform nodes to parallax space.
    p_nodes = inverse_transform(np.sqrt(2)*sigma[:,np.newaxis]*nodes[np.newaxis,:] + root_z[:,np.newaxis],
                                transform=kwargs['transform'], b=b)

    # Get main terms in
    beta, pi_mu, pi_err, n = args
    integral = integrand(p_nodes, beta[:,np.newaxis], pi_mu[:,np.newaxis], pi_err[:,np.newaxis], n, b=b) \
                    * 1/jacobian(p_nodes, transform=kwargs['transform'], b=b)

    # Correction for Gauss-Hermite Gaussian multiplier
    correction = np.sqrt(2)*sigma[:,np.newaxis] / np.exp(-nodes**2)

    return np.sum(weights[np.newaxis,:]*integral*correction, axis=1)


def integrate_gh_gap(integrand, z_mode, sigma, args, transform='none', a=None, b=None, degree=10):

    if a is not None: a=a[:,np.newaxis]
    if b is not None: b=b[:,np.newaxis]

    # Draw nodes and weights for Gauss-Hermite.
    nodes, weights = np.polynomial.hermite.hermgauss(degree)
    # Transform nodes to parallax space.
    p_nodes = trans_i(np.sqrt(2)*sigma[:,np.newaxis]*nodes[np.newaxis,:] + z_mode[:,np.newaxis],
                                transform=transform, b=b, a=a)

    args = [np.repeat([arg,], degree, axis=0).T for arg in args]
    integral = integrand(p_nodes, *args)/jac(p_nodes, transform=transform, b=b, a=a)

    # Correction for Gauss-Hermite Gaussian multiplier
    correction = np.sqrt(2)*sigma[:,np.newaxis] / np.exp(-nodes**2)

    return np.sum(weights[np.newaxis,:]*integral*correction, axis=1)

def d2logIJ_dp2_plbrk(p, beta, pi_mu, pi_err, n, transform='none', b=None):
    if transform=='none': return -n/p**2 - 2*beta/p**3 - 1/pi_err**2
    elif transform=='log': return -(n+1)/p**2 - 2*beta/p**3 - 1/pi_err**2
    elif transform=='logit': return -1/(b-p)**2-(n+1)/p**2 - 2*beta/p**3 - 1/pi_err**2


# Root finding algorithms

@njit
def bisect_algo(foo, a, b, tol=1e-10, args=None):
    x0,x2 = (a,b)
    fx0 = foo(x0, args)
    fx2 = foo(x2, args)
    x1 = (x0+x2)/2
    fx1 = foo(x1, args)

    px0,px1,px2=(x0,x1,x2)

    while abs(fx1)>tol:

        x1 = (x0+x2)/2
        fx1 = foo(x1, args)

        if fx1*fx0 < 0:
            x2=x1
        elif fx1*fx0 > 0:
            x0=x1

        if (x0==px0)&(x1==px1)&(x2==px2):
            if (x2-x0)/x0<1e-5:
                return x1
            else:
                print('Bad: ', x0, x2, x1, fx0, fx2)
                raise ValueError('No convergence...')
        px0,px1,px2=(x0,x1,x2)

    return x1

#@njit
def ridders_algo(foo, a, b, tol=1e-5, args=None):

    x0,x2 = (a,b)
    fx0 = foo(x0, args)
    fx2 = foo(x2, args)
    x3 = (x0+x2)/2
    fx3 = foo(x3, args)

    px0,px1,px2,px3=(x0,x3,x2,x3)

    while abs(fx3)>tol:

        x1 = (x0+x2)/2
        fx1 = foo(x1, args)

        x3 = x1 + (x1-x0)*(np.sign(fx0)*fx1/np.sqrt(fx1**2 +- fx0*fx2))
        fx3 = foo(x3, args)

        if fx3*fx1 < 0:
            if x1<x3:
                x0=x1
                fx0=fx1
                x2=x3
                fx2=fx3
            elif x1>x3:
                x0=x3
                fx0=fx3
                x2=x1
                fx2=fx1
        elif fx3*fx1 > 0:
            if fx3*fx0<0:
                x2=x3
                fx2=fx3
            elif fx3*fx2<0:
                x0=x3
                fx0=fx3
            elif fx0*fx2>0:
                raise ValueError('bounds have same sign...')

        if (x0==px0)&(x1==px1)&(x2==px2)&(x3==px3):
            if (x2-x0)/x0<1e-5:
                #print(fx0,fx2)
                return x3
            else:
                print(x0, x2, fx0, fx2)
                raise ValueError('No convergence...')
        px0,px1,px2,px3=(x0,x1,x2,x3)

    return x3

@njit
def get_polyroots(coeffs, roots, b=None):

    for i in range(coeffs.shape[0]):
        if False: roots[i] = ridders_algo(polynomial_jit, 0., b[i], args=(coeffs[i,:]))
        if True: roots[i] = bisect_algo(polynomial_jit, 0., b[i], args=(coeffs[i,:]))

    return roots

@njit
def get_fooroots(func, roots, a=None, b=None, args=None):

    for i in range(len(args[0])):
        args_i = [arg[i] for arg in args]
        if a is None:
            if False: roots[i] = ridders_algo(func, 0., b[i], args=args_i)
            if True: roots[i]  = bisect_algo(func, 0., b[i], args=args_i)
        else:
            if False: roots[i] = ridders_algo(func, b[i], a[i], args=args_i)
            if True: roots[i]  = bisect_algo(func, b[i], a[i], args=args_i)

    return roots

@njit
def get_roots_ridder_hm(coeffs, b=None):
    # hm stands for homemade
    roots = np.zeros(coeffs.shape[0]) + np.nan

    for i in range(coeffs.shape[0]):
        if False: roots[i] = ridders_algo(polynomial_jit, 0., b[i], args=(coeffs[i,:]))
        if True: roots[i] = bisect_algo(polynomial_jit, 0., b[i], args=(coeffs[i,:]))
        #if roots[i]==0.: roots[i] = bisect_algo(polynomial_jit, 0., b[i], args=(coeffs[i,:]))

    return roots

#@njit
def get_fooroots_ridder_hm(func, b=None, a=None, args=None):
    # hm stands for homemade
    roots = np.zeros(len(args[0])) + np.nan
    for i in range(len(args[0])):
        args_i = tuple([arg[i] for arg in args])
        if a is None:
            if False: roots[i] = ridders_algo(func, 0., b[i], args=args_i)
            if True: roots[i]=bisect_algo(func, 0., b[i], args=args_i)
        else:
            if False: roots[i] = ridders_algo(func, a[i], b[i], args=args_i)
            if True: roots[i]=bisect_algo(func, a[i], b[i], args=args_i)

    return roots

@njit
def polynomial_jit(p,coeffs):
    ans=0
    for i in range(len(coeffs)):
        ans += coeffs[i]*p**(len(coeffs)-i-1)
    return ans
