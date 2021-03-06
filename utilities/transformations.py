
import numpy as np

def Mbol_func(m, a=4):
    return -2.5*a*np.log(m)/np.log(10)


def logit(p, pmin=0, pmax=1):
    return np.log((p-pmin)/(pmax-p))
def expit(x, pmin=0, pmax=1):
    return (pmax*np.exp(x)+pmin)/(np.exp(x)+1)


def logit_scaled(a, b):
    return (lambda x:(a + b*np.exp(x))/(1+np.exp(x)), lambda x:np.log((x-a)/(b-x)), lambda x:(x-a)*(b-x)/(b-a))

func_inv_jac = {'none': lambda a, b: (lambda p:p, lambda x:x, lambda x:1),
                'exp':  lambda a, b: (lambda p:np.exp(p), lambda x:np.log(x), lambda x:x),
                'expn': lambda a, b: (lambda p:np.exp(-p), lambda x:-np.log(x), lambda x:x),
                'nexp': lambda a, b: (lambda p:-np.exp(p), lambda x:np.log(-x), lambda x:x),
                'logit':lambda a, b: (lambda p: (np.exp(p))/(1+np.exp(p)), lambda x: np.log(x/(1-x)), lambda x:x*(1-x)),
                'logit_scaled':lambda a, b: (lambda p:(a + b*np.exp(p))/(1+np.exp(p)), lambda x:np.log((x-a)/(b-x)), lambda x:(x-a)*(b-x)/(b-a))}

def logit_label(s,a,b):
    if (a==0) and (b==1): return r'logit$({0})$'.format(s.replace('$',''))
    elif (a==0): return r'logit$({0}/{1})$'.format(s.replace('$',''),b)
    else: return r'logit$(\frac{{ {0}-{1} }}{{ {2}-{1} }} )$'.format(s.replace('$',''),a,b)
    #else: return r'logit$(({0}-{1})/({2}-{1}))$'.format(s.replace('$',''),a,b)
func_labels = {'none': lambda s, a, b: s,
                'exp':  lambda s, a, b: r'$\log({0})$'.format(s.replace('$','')),
                'expn': lambda s, a, b: r'$-\log({0})$'.format(s.replace('$','')),
                'nexp': lambda s, a, b: r'$\log(-{0})$'.format(s.replace('$','')),
                'logit':lambda s, a, b: r'logit$({0})$'.format(s.replace('$','')),
                'logit_scaled':logit_label}#ambda s, a, b: r'$logit(({0}-{1})/({2}-{1}))$'.format(s.replace('$',''),a,b)}

label_dict = {'alpha1':r'$\alpha_1$', 'alpha2':r'$\alpha_2$', 'alpha3':r'$\alpha_3$',
              'fD': r'$f_D$', 'hz':r'$h_z$', 'w':r'$w$'}
label_set = [r'$w$', r'$h_\mathrm{Tn}$', r'$f_D$',
             r'$w$', r'$h_\mathrm{Tk}$', r'$f_D$',
             r'$w$', r'$n_\mathrm{H}$', r'$f_D$',
             r'$\alpha_1$', r'$\alpha_2$']
