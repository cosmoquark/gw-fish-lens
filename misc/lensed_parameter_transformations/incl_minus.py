import numpy as np
def incl_minus(incl, PhiL, alphaHatMinus, thetaMinus=0, beta=0):
    return np.arccos(np.cos(incl) * (np.cos((alphaHatMinus + (-beta) + (-thetaMinus))) + (np.cos(PhiL) * np.sin(incl) * (((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))**(-1/2)) * np.sin((alphaHatMinus + (-beta) + (-thetaMinus))))))