import numpy as np
def incl_plus(incl, PhiL, alphaHatPlus, thetaPlus=0, beta=0):
    return np.arccos(np.cos(incl) * (np.cos((alphaHatPlus + beta + (-thetaPlus))) + (-(np.cos(PhiL) * np.sin(incl) * (((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))**(-1/2)) * np.sin((alphaHatPlus + beta + (-thetaPlus)))))))