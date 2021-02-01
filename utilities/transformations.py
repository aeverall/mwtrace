
import numpy as np

def Mbol_func(m, a=4):
    return -2.5*a*np.log(m)/np.log(10)


def logit(p, pmin=0, pmax=1):
    return np.log((p-pmin)/(pmax-p))
def expit(x, pmin=0, pmax=1):
    return (pmax*np.exp(x)+pmin)/(np.exp(x)+1)
