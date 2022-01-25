import math as m
import numpy as np
import matplotlib.pyplot as plt
from rsa import sign
from scipy import signal


def gen_fc_fl_fu(_fmax, _fmin, _ratio):
    
    n_octs = m.log2(_fmax/_fmin)
    _n_bands = m.ceil(n_octs/_ratio) + 1
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
    _sos = signal.butter(_order, [lower, upper], btype='bandpass', output='sos') # obtain sos matrix

    if _display:
        h, f = signal.sosfreqz(_sos,worN=1024, fs=_fs)
        plt.semilogx(h[1:], np.array([20 * m.log10(abs(value)) for value in f[1:]]), label=None)
        plt.ylim(-120, 20)
        
    return _sos


def plot_bins(center_fqs, spl):
    
    spl_pad = np.zeros(len(spl))
    fcs = center_fqs
    fig = plt.figure(figsize = (15, 10))

    for i in range(len(spl_pad)):

        start = min(spl)
        stop = spl[i]

        if start < 0 and stop < 0:
            spl_pad[i] = abs(start - stop)
        if start < 0 and stop >= 0:
            spl_pad[i] = abs(start) + stop
        if start >= 0:
            spl_pad[i] = stop - start

    # padding for full display
    spl_pad = np.append(spl_pad, 0)
    fcs = np.append(fcs, 0)
    
    # https://stackoverflow.com/questions/44068435/setting-both-axes-logarithmic-in-bar-plot-matploblib
    plt.bar(np.array(fcs)[:-1],                         \
        [spl_pad[i] for i in range(len(spl_pad))][:-1], \
        bottom=start,                                   \
        width=np.diff(fcs),                             \
        log=True,                                       \
        ec="k",                                         \
        align="edge")

    plt.xscale("log")
    plt.yscale("linear")
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')
    plt.xlabel("f")
    plt.ylabel("dB SPL")
    plt.show()


def plot_bode(center_fqs, spl):
    # not mine
    fig, ax = plt.subplots()
    ax.semilogx(center_fqs, spl, 'b')
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel('Level [dB]')
    plt.xlim(11, 25000)
    ax.set_xticks([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(['16', '31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
    plt.show()


def oct_filterbank(x, n_bands, dec_factor, dec_iir_ord, fs, fls, fus, display=False, window_size=1024):

    if int(len(x) / window_size) * window_size < len(x):
        pad = np.full(window_size - (len(x) % window_size), 0)

    elif len(x) < window_size:
        pad = np.full(window_size - (len(x)), 0)

    else:
        pad = []

    x = np.append(x, pad)
    n_frames = int(len(x)/window_size)
    oct_features = np.zeros((n_frames, n_bands))

    for frame in range(n_frames):

        frame_buf = x[frame * window_size:(frame + 1) * window_size]
        spl = np.zeros(n_bands)
        for band in range(n_bands):
            if dec_factor[band] == 1:
                x_dec = frame_buf
            elif dec_factor[band] <= 8:
                x_dec = signal.decimate(frame_buf, dec_factor[band], n=dec_iir_ord)
            else:
                x_dec = signal.decimate(frame_buf, int(dec_factor[band]/(dec_factor[band]/8)), n=dec_iir_ord)
                x_dec = signal.decimate(x_dec, int(dec_factor[band]/8), n=dec_iir_ord)
            cur_sos = gen_bandpass(_fs=int(fs/dec_factor[band]), _fl=fls[band], _fu=fus[band], _display=display)
            sig_out = signal.sosfilt(cur_sos, x_dec)
            ms = rms(sig_out, root=False)
            #spl[band] = 10 * np.log10(ms / 2e-5)
            spl[band] = 20 * np.log10(np.std(sig_out) / 2e-5)

        oct_features[frame] = spl

    return oct_features


# be aware that this runs over in case of val not being between 0 and 1/-1
def rms(buffer, root=True):
    b_sum = 0
    for val in buffer:
        b_sum = b_sum + val ** 2
    rms = b_sum / (len(buffer))

    if(root):
        rms = np.sqrt(rms)

    return rms
