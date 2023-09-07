import numpy as np
from datetime import datetime
import scipy.stats
import scipy.integrate
import pylandau

def gauss(x, mu, sigma):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))

def multigauss(x, *params):
    if len(params) % 3 != 0:
        raise ValueError("Wrong number of parameters!")
    
    amps = np.array(params[0::3])[:, np.newaxis]
    mus = np.array(params[1::3])[:, np.newaxis]
    sigmas = np.array(params[2::3])[:, np.newaxis]


    y = amps * gauss(x, mus, sigmas)

    return y.sum(axis=0), y


def sps_poisson(x, ped = 50, gain = 50, width_base = 5, width_scale = 5, poisson_k = 5, ped_offset = 0, output_single_peaks = False):
    n_peaks = int(3*poisson_k)+5

    amps = scipy.stats.poisson.pmf(np.arange(0, n_peaks, 1), poisson_k)
    mus  = [ped,  *(ped - ped_offset + gain * np.arange(1, n_peaks, 1))]
    sigmas = np.sqrt(width_base**2 +  width_scale**2 * np.arange(0, n_peaks, 1))

    params = np.empty((n_peaks*3,), dtype=amps.dtype)
    params[0::3] = amps
    params[1::3] = mus
    params[2::3] = sigmas

    if output_single_peaks:
        return multigauss(x, *params)
    else:
        return multigauss(x, *params)[0]
    

def sps_landau_poisson(x, ped_x = 50, ped_a = 1, ped_w = 4, gain = 50, width_base = 5, width_scale = 5, landau_mpv = 10, landau_sigma = 0.15, ped_offset = 0, output_single_peaks = False):
    n_peaks = int(5*landau_mpv)+5

    conv_iterations = 100
    x_landau = np.linspace(0, 10*landau_mpv, conv_iterations)
    x_step = x_landau[1] - x_landau[0]
    landau = pylandau.landau_pdf(x_landau, landau_mpv, landau_sigma*landau_mpv)

    #print(scipy.integrate.quad(lambda xxx: pylandau.get_landau_pdf(xxx, landau_mpv, landau_sigma*landau_mpv), 0, 10*landau_mpv))


    amps = np.zeros(n_peaks, dtype=np.float64)

    for xl, l in zip(x_landau, landau):
        scale = scipy.integrate.quad(lambda xxx: pylandau.get_landau_pdf(xxx, landau_mpv, landau_sigma*landau_mpv), xl-x_step/2, xl+x_step/2)[0]
        amps += scipy.stats.poisson.pmf(np.arange(0, n_peaks, 1), xl) * scale

    amps[0] += ped_a

    mus  = [ped_x,  *(ped_x - ped_offset + gain * np.arange(1, n_peaks, 1))]
    sigmas = np.sqrt(width_base**2 +  width_scale**2 * np.arange(0, n_peaks, 1))
    sigmas[0] = ped_w

    params = np.empty((n_peaks*3,), dtype=amps.dtype)
    params[0::3] = amps
    params[1::3] = mus
    params[2::3] = sigmas

    
    if output_single_peaks:
        return multigauss(x, *params)
    else:
        return multigauss(x, *params)[0]

