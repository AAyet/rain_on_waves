#############################################
# Code used to compute volume fraction of air
# in the presence of bubbles due to rain.
#############################################

import numpy as np
from scipy.integrate import cumtrapz, trapz


#parameters
g = 9.8 #gravity

mu_w = 10**(-3) #dynamic viscosity of the water
mu_a = 1.8 * 10**(-5) #dynamic viscosity of the air-
#mu_a = 1.8 * 10**(-4) #dynamic viscosity of the air-


nu_w = 9.79e-7
nu_a = 1.51e-5
#nu_a = 1.51e-4


rho_a = 1. #density of air
rho_w = 1000. #density of water

gamma = 1.4 # ratio of specific heats for air bubble
P0 = 1.01e5 # surface pressure

sigma = 72.8 * 10**-3 # surface water tension 


def w_t(r, g = g, mu = mu_w, rho_w = rho_w):
    """
    Terminal velocity of a raindrops following Dingle and Lee 1972
    """
    return ( (r <= 0.7) * ((-17.8951 + 448.9498 * (2 * r) + 16.3719 * (2 * r)**2 - 45.9516 * (2*r)**3)) + 
             (r > 0.7) * (24.1660 + 448.8336 * (2*r) - 75.6265 * (2 * r)**2 + 4.2659 * (2 * r)**3)
            ) * 0.01


def w_b(r, g = g, mu = mu_w, rho_w = rho_w):
    """
    Terminal velocity of a bubble following clift1978
    """
    #return ( 2.14 * sigma / (rho_w * r) + 0.505 * g  * 2 * r)**(0.5) *  10**-2
    return ( sigma  *  10**7 / (rho_w * r) +  g * r * 10)**(0.5) *  10**-2


def MP(r, R):
    """
    Marshall Palmer distribution of the number of drops of a given radius r
    for a given rain intensity R
    """
    
    return 2 * 8000 * np.exp(-R**(-0.21) * r * 8.2)


def DRD(r, R, mu_a, g, rho_w):
    '''
    Drop density distribution
    '''

    return w_t(r) * MP(r,R) 


def f(r):
    '''
    Inversion law to go from drop radius to bubble radius 
    following medwin
    '''

    return ((r >= 1.1)  * 3.25 * (160./(8*r**3) +0.6)**(-1) + 
           (r <= 0.55) * (r >= 0.41) * 0.22 + 
           (r < 0.41) * 0 +
           (r > 0.55) * (r < 1.1) * 0
           )


def percen(r):
    '''
    percentage of bubble production by a drop of radius r
    
    the drop radius is in mm
    '''
    return (r > 1.1) * np.maximum(0, np.polyval(np.load('drop.npy'), 2 * r)
                                           ) + (r <= 1.1) 


def N(r, R, mu_a, mu_w,  g, rho_w, mod):
    '''
    Density of bubbles given a radius of drops
    
    mod is the modulation of the terminal velocity of bubbles (called c_e in the paper)
    '''
    
    perc = percen(r)
    
    DR = DRD(r, R, mu_a, g, rho_w)
    r_b = f(r)
    if r_b != 0:
    
        w = mod * w_b(r_b, g, mu_w, rho_w)
    
        return perc * DR / w
    
    else: return 0.

