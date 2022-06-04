# ------[Octafilt3r]------
#   @ name: octafilter.weightings
#   @ auth: Jakob Tschavoll
#   @ vers: 0.1
#   @ date: 2022

"""
Normed or specific weightings for filterbanks and audio signals
"""

import numpy as np
from numpy import pi
from scipy.signal import zpk2tf, zpk2sos, freqs, sosfilt, bilinear_zpk

__all__ = ['A_weight', 'plot_bins']

def _A_weighting():
    """
    Calculate zeros, poles and gain for A-weighting as specified by norm

    Params
    ------
    None

    Returns
    -------
    `z`:    zeros
    `p`:    poles
    `k`:    gain
    """
    # redux from https://github.com/endolith/waveform_analysis/blob/master/waveform_analysis/weighting_filters/ABC_weighting.py

    z = [0, 0]
    p = [-2*pi*20.598997057568145,
         -2*pi*20.598997057568145,
         -2*pi*12194.21714799801,
         -2*pi*12194.21714799801]
    k = 1

    p.append(-2*pi*107.65264864304628)
    p.append(-2*pi*737.8622307362899)
    z.append(0)
    z.append(0)

    b, a = zpk2tf(z, p, k)
    k /= abs(freqs(b, a, [2*pi*1000])[1][0])

    return np.array(z), np.array(p), k



def _A_sos(fs):
    """
    Convert zpk-form into sos-matrix

    Params
    ------
    `fs`:       desired sampling frequency

    Returns
    -------
    `zpk_sos`:  A-weighting in sos-shape
    """
    # redux from https://github.com/endolith/waveform_analysis/blob/master/waveform_analysis/weighting_filters/ABC_weighting.py

    z, p, k = _A_weighting()
    z_d, p_d, k_d = bilinear_zpk(z, p, k, fs)
    return zpk2sos(z_d, p_d, k_d)


def A_weight(x, fs):
    """
    A-weight a given signal

    Params
    ------
    `x`:    signal in array shape
    `fs`:   desired sampling frequency

    Returns
    -------
    `y`:  A-weighted signal
    """
    # redux from https://github.com/endolith/waveform_analysis/blob/master/waveform_analysis/weighting_filters/ABC_weighting.py

    return sosfilt(_A_sos(fs), x)


def _ICS43432_correction_sos(fs=48000):
    # coefficients from https://github.com/ikostoski/esp32-i2s-slm/blob/master/math/ics43432.m

    supported_fs = [32000, 48000]

    if fs not in supported_fs:
        print(f"Octafilt3r error: sampling rate <{fs}> no supported!")
        return None

    if fs == 32000:
        sos_real = np.asarray([[2.66513655490457,   -5.22423262968907,  2.56000810375323,   1,  0,                 0                    ],
                            [   1,                  -0.391430874040863, -0.0494378769080195,1,  1.51216111792406,  0.572836798890286    ],
                            [   1,                  0.435508091933108,  0.536629986251847,  1,  -1.96541197268139, 0.966525580593619    ]])

        sos_cor = np.asarray([[ 0.375215295501361,  0,                  0,                  1,  -0.391430874040863, -0.0494378769080195 ],
                            [   1,                  -1.96541197268139,  0.966525580593621,  1,  -1.96021198991664,  0.96055419713565    ],
                            [   1,                  1.51216111792406,   0.572836798890284,  1,  0.435508091933108,  0.536629986251847   ]])

    if fs == 48000:    
        sos_real = np.asarray([[-2.18657127359417,  2.51126290286187,   -0.329296622151638, 1,  -0.544048068678254, -0.248361793539276  ],
                            [   1,                  -1.79028576859208,  0.804085859544701,  1,  -1.90991179395606,  0.910830218057788   ],
                            [   1,                  -0.403298902792304, 0.207346184247794,  1,  0,                  0                   ]])

        sos_cor = np.asarray([[ -0.457337024443869, 0.316723378796653,  0.133668934632936,  1,  -1.14849350359112,  0.150599537334247   ],
                            [   1,                  0,                  0,                  1,  -0.403298902792304, 0.207346184247794   ],
                            [   1,                  -1.76142162088186,  0.773977067696167,  1,  -1.79028576859208,  0.804085859544701   ]])

    return sos_cor, sos_real


def ICS43432_correction(x, fs=48000):

    return sosfilt(_ICS43432_correction_sos(fs), x)
