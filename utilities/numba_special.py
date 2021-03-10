from numba.extending import get_cython_function_address
from numba import vectorize, njit
import ctypes
import numpy as np


addr = get_cython_function_address("scipy.special.cython_special", "beta")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
beta_fn = functype(addr)
@vectorize('float64(float64,float64)')
def vec_beta(x,y):
    return beta_fn(x,y)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
gammaln_fn = functype(addr)
@vectorize('float64(float64,float64)')
def vec_gammaln(x,y):
    return gammaln_fn(x,y)

@vectorize('float64(float64,float64)')
def vec_gamma(x,y):
    return np.exp(gammaln_fn(x,y))

addr = get_cython_function_address("scipy.special.cython_special", "gammainc")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
gammainc_fn = functype(addr)
@vectorize('float64(float64, float64)')
def vec_gammainc(x, y):
    return gammainc_fn(x, y)

addr = get_cython_function_address("scipy.special.cython_special", "gammaincc")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
gammaincc_fn = functype(addr)
@vectorize('float64(float64, float64)')
def vec_gammaincc(x, y):
    return gammaincc_fn(x, y)

# addr = get_cython_function_address("scipy.special.cython_special", "hyp2f1")
# functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
# gammaincc_fn = functype(addr)
# @vectorize('float64(float64, float64)')
# def vec_gammaincc(x, y):
#     return gammaincc_fn(x, y)


# addr = get_cython_function_address("scipy.special.cython_special", "logsumexp")
# functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
# logsumexp_fn = functype(addr)
# @vectorize('float64(float64, float64)')
# def vec_logsumexp(x, y):
#     return logsumexp_fn(x, axis=0, b=y)
