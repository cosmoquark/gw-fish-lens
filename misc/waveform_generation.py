# Import pycbc waveform and beam pattern functions
from pycbc import waveform, psd
from pycbc.detector import Detector, get_available_detectors
import pylab as plt
import numpy as np

def create_whitened_waveform(phase=np.pi/2, inclination=np.pi/2, polarization=0.1, t_gps = 1126259462.0, mass1=80, mass2=80, spin1x=0, spin1y=0, spin1z=0, spin2x=0, spin2y=0, spin2z=0, approximant = "SEOBNRv4PHM"):
    # Create a whitened waveform in time domain
    # phase = np.pi/2
    # inclination = np.pi/2
    # hp, hc = waveform.get_td_waveform(approximant="IMRPhenomXPHM", mass1=80, mass2=80, delta_t=1.0/4096, f_lower=10, distance=2000, inclination=inclination, coa_phase=phase)
    hp, hc = waveform.get_td_waveform(approximant=approximant, mass1=mass1, mass2=mass2, delta_t=1.0/4096, f_lower=10, distance=2000, inclination=inclination, coa_phase=phase, 
                                      spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, spin2x=spin2x, spin2y=spin2y, spin2z=spin2z)
    # Set the hp and hc start time to the GPS time
    hp.start_time = hc.start_time = t_gps
    # Compute the plus and cross polarizations
    detector = Detector("L1")
    right_ascension = 0.1
    declination = 0.15
    fp, fc = detector.antenna_pattern(right_ascension, declination, polarization, t_gps)
    h = fp * hp + fc * hc
    # Whiten the waveform
    h_freq = h.to_frequencyseries()
    length = len(h_freq)
    delta_f = h_freq.delta_f
    low_freq_cutoff = h_freq.sample_frequencies[0]
    L1_psd = psd.aLIGOZeroDetHighPower(length, delta_f, low_freq_cutoff)
    L1_asd = np.sqrt(L1_psd)
    h_freq_whitened = h_freq / L1_asd
    h_freq_whitened.data[np.isfinite(h_freq_whitened.data)==False] = 0 # Set NaNs and infs to zero
    h_whitened = h_freq_whitened.to_timeseries()
    return h_whitened