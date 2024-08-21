# Lensed waveform parameters

import numpy as np

def incl_plus_approx(incl, PhiL, alphahat, beta=0, theta=0):
    return np.acos(
        np.cos(incl)
        - # negative in plus case
        ( np.sin(incl) * np.cos(incl) * np.cos(PhiL) )
        / np.sqrt( np.sin(incl)**2 * np.sin(PhiL)**2 + np.cos(incl)**2 )
        * (alphahat + beta - theta) # + beta in plus case
    )


def incl_minus_approx(incl, PhiL, alphahat, beta=0, theta=0):
    return np.acos(
        np.cos(incl)
        + # positive in minus case
        ( np.sin(incl) * np.cos(incl) * np.cos(PhiL) )
        / np.sqrt( np.sin(incl)**2 * np.sin(PhiL)**2 + np.cos(incl)**2 )
        * (alphahat - beta - theta) # - beta in minus case
    )

def Phi_coal_plus_approx(Phi_coal, incl, PhiL, alphahat, beta=0, theta=0):
    return np.acos(
        np.cos(PhiL)
        + # positive in plus
        ( np.sin(incl)**(-2) * np.sin(Phi_coal)  *np.sin(PhiL) )
        / np.sqrt( np.tan(incl)**(-2) + np.sin(PhiL)**2 )
        * (alphahat + beta - theta) # + beta in plus case
    )

def Phi_coal_minus_approx(Phi_coal, incl, PhiL, alphahat, beta=0, theta=0):
    return np.acos(
        np.cos(PhiL)
        - # negative in minus case
        ( np.sin(incl)**(-2) * np.sin(Phi_coal)  *np.sin(PhiL) )
        / np.sqrt( np.tan(incl)**(-2) + np.sin(PhiL)**2 )
        * (alphahat - beta - theta) # - beta in minus case
    )

def polarization_angle_plus_approx(pol_ang, incl, PhiL, alphahat, beta=0, theta=0):
    return np.acos(
        np.cos(pol_ang)
        - # negative in plus case
        ( np.tan(incl) * np.sin(PhiL)  *np.sin(pol_ang) )
        / np.sqrt( np.sin(incl)**2 * np.sin(PhiL)**2 +np.cos(incl)**2 )
        * (alphahat + beta - theta) # + beta in plus case
    )

def polarization_angle_minus_approx(pol_ang, incl, PhiL, alphahat, beta=0, theta=0):
    return np.acos(
        np.cos(pol_ang)
        + # positive in minus case
        ( np.tan(incl) * np.sin(PhiL)  *np.sin(pol_ang) )
        / np.sqrt( np.sin(incl)**2 * np.sin(PhiL)**2 +np.cos(incl)**2 )
        * (alphahat - beta - theta) # - beta in minus case
    )

