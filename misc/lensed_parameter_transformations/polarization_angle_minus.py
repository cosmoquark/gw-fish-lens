import numpy as np
def polarization_angle_minus(psi, incl, PhiL, alphaHatMinus, thetaMinus=0, beta=0):
    return np.arccos((((np.cos(psi) * (1/((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))) * (((np.cos(incl)**2) * ((-((np.cos(PhiL)**2) * (-1 + np.cos((alphaHatMinus + (-beta) + (-thetaMinus)))))) + np.cos((alphaHatMinus + (-beta) + (-thetaMinus))))) + (np.cos((alphaHatMinus + (-beta) + (-thetaMinus))) * (np.sin(incl)**2) * (np.sin(PhiL)**2)))) + (-8 * np.cos(incl) * (1/(6 + (2 * np.cos((2 * incl))) + np.cos(((2 * incl) + (-2 * PhiL))) + (-2 * np.cos((2 * PhiL))) + np.cos((2 * (incl + PhiL))))) * np.sin((2 * PhiL)) * np.sin(psi) * (np.sin(((1/2) * (alphaHatMinus + (-beta) + (-thetaMinus))))**2))) * ((np.cos((alphaHatMinus + (-beta) + (-thetaMinus))) * np.sin(incl)) + (-((np.cos(incl)**2) * np.cos(PhiL) * (((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))**(-1/2)) * np.sin((alphaHatMinus + (-beta) + (-thetaMinus)))))) * ((((np.sin(PhiL)**2) * (1/((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))) * (np.sin((alphaHatMinus + (-beta) + (-thetaMinus)))**2)) + (((np.cos((alphaHatMinus + (-beta) + (-thetaMinus))) * np.sin(incl)) + (-((np.cos(incl)**2) * np.cos(PhiL) * (((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))**(-1/2)) * np.sin((alphaHatMinus + (-beta) + (-thetaMinus))))))**2))**(-1/2))) + (np.sin(PhiL) * np.sin((alphaHatMinus + (-beta) + (-thetaMinus))) * ((((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2))) * (((np.sin(PhiL)**2) * (1/((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))) * (np.sin((alphaHatMinus + (-beta) + (-thetaMinus)))**2)) + (((np.cos((alphaHatMinus + (-beta) + (-thetaMinus))) * np.sin(incl)) + (-((np.cos(incl)**2) * np.cos(PhiL) * (((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))**(-1/2)) * np.sin((alphaHatMinus + (-beta) + (-thetaMinus))))))**2)))**(-1/2)) * ((np.cos(incl) * np.sin(incl) * (((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))**(-3/2)) * np.sin(psi) * ((2 * np.sin(incl) * (np.sin(PhiL)**2) * (((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))**(1/2)) * (np.sin(((1/2) * (alphaHatMinus + (-beta) + (-thetaMinus))))**2)) + ((np.cos(incl)**2) * np.cos(PhiL) * np.sin((alphaHatMinus + (-beta) + (-thetaMinus)))) + (np.cos(PhiL) * (np.sin(incl)**2) * (np.sin(PhiL)**2) * np.sin((alphaHatMinus + (-beta) + (-thetaMinus)))))) + (np.cos(psi) * ((-((-1 + np.cos((alphaHatMinus + (-beta) + (-thetaMinus)))) * ((1/np.tan(PhiL))**3))) + (-2 * (1/np.tan(PhiL)) * ((1/np.sin(PhiL))**2) * (np.sin(((1/2) * (alphaHatMinus + (-beta) + (-thetaMinus))))**2)) + ((1/np.sin(PhiL)) * (1/np.cos(incl)) * (((np.cos(incl)**2) + ((np.sin(incl)**2) * (np.sin(PhiL)**2)))**(1/2)) * np.sin((alphaHatMinus + (-beta) + (-thetaMinus))) * np.tan(incl))) * (1/(((1/np.sin(PhiL))**2) + (np.tan(incl)**2)))) + (np.cos(incl) * np.sin(psi) * (1/(((1/np.sin(PhiL))**2) + (np.tan(incl)**2))) * (((-1 + np.cos((alphaHatMinus + (-beta) + (-thetaMinus)))) * ((1/np.tan(PhiL))**2)) + ((1/np.sin(PhiL))**2) + (np.cos((alphaHatMinus + (-beta) + (-thetaMinus))) * (np.tan(incl)**2)))))))