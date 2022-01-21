import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def gen_fc_fl_fu(_fmax, _fmin, _ratio):
    
    n_octs = m.log2(_fmax/_fmin)
    _n_bands = m.ceil(n_octs/_ratio)
    _fcs = np.array([_fmin * 2. ** (i * _ratio) for i in range(0, _n_bands)])
    _fls = np.array([cur_fc * 2. ** ((-1) * _ratio / 2) for cur_fc in _fcs])
    _fus = np.array([cur_fc * 2. ** (_ratio / 2) for cur_fc in _fcs])

    # turn the arrays around to start with the upper bands
    _fcs = _fcs[::-1] 
    _fls = _fls[::-1]
    _fus = _fus[::-1]

    return _fcs, _fls, _fus, _n_bands


def gen_bandpass(_fs, _fl, _fu, _order=6, _display=False):
    ny = 1/2 * _fs
    lower = _fl / ny
    upper = _fu / ny
    _sos = signal.butter(_order, [lower, upper], btype='band', output='sos') # obtain sos matrix

    if _display:
        h, f = signal.sosfreqz(_sos,worN=1024, fs=_fs)
        plt.semilogx(h[1:], np.array([20 * m.log10(abs(value)) for value in f[1:]]), label=None)
        
    return _sos