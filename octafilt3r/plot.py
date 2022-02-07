# ------[Octafilt3r]------
#   @ name: octafilter.plot
#   @ auth: Jakob Tschavoll
#   @ vers: 0.1
#   @ date: 2022

"""
Octave-focused plot and display functions for filters in tandem with `octafilt3r.filter`
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from octafilt3r import filter as o3f

__all__ = ['oct_spectrogram', 'plot_bins']


def oct_spectrogram(features, fs, frame_size, fmax=20000, fmin=20, ratio=1/3):
    """
    Draw a spectrogram of `dBFS`-levels derived from octave filters.

    Params
    ------
    `features`:     Matrix of `dBFS`-levels over time and frequency. Must be of shape `(n_frames, n_bands)`.
    `fs`:           Sample rate of original signal.
    `frame_size`:   Width of sample-buffers which were analyzed in one timestep in `rolling_oct_bank`.

    Returns
    -------
    None (display function)
    """
    
    fps = int(fs / frame_size)
    fcs, fls, fus, n_bands = o3f._gen_fc_fl_fu(fmax, fmin, ratio)

    frame2s = []
    xlabels = []

    bin2freq = np.arange(n_bands)
    ylabels = []

    for i in range(len(fcs)):
        ylabels.append(str(int(fcs[i])))
    
    feat = np.transpose(features)
    frames = len(feat[0])

    for i in range(int(np.ceil(frames / fps)) + 1):
        frame2s.append(fps * i)
        xlabels.append(str(int((fps * i) / fps)))

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.pcolormesh(feat, cmap = 'rainbow')
    plt.title('octave based spectrogram')
    plt.xlabel("s")
    plt.ylabel("Hz")
    ax.set_xticks(frame2s)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(bin2freq)
    ax.set_yticklabels(ylabels)
    plt.colorbar(label='dBFS')


def plot_bins(fcs, lvl):
    """
    Plot logarithmic bins of levels of a time instance in style of SPL-meters.

    Params
    ------
    `fcs`:  Center frequencies of all bands.
    `lvl`:  Detected level (in `dBFS`) from output of filterbank

    Returns
    -------
    None (display function)
    """
    
    spl_pad = np.zeros(len(lvl))
    fig = plt.figure(figsize = (10, 5))

    for i in range(len(spl_pad)):

        start = min(lvl) - 10
        stop = lvl[i]

        if start < 0 and stop < 0:
            spl_pad[i] = abs(start - stop)
        if start < 0 and stop >= 0:
            spl_pad[i] = abs(start) + stop
        if start >= 0:
            spl_pad[i] = stop - start

    # padding for full display
    spl_pad = np.append(spl_pad, 0)
    fcs = np.append(fcs, fcs[-1] * 6/5)
    
    # https://stackoverflow.com/questions/44068435/setting-both-axes-logarithmic-in-bar-plot-matploblib
    plt.bar(np.array(fcs)[:-1],                         \
        np.array(spl_pad)[:-1],                         \
        bottom=start,                                   \
        width=np.diff(fcs),                             \
        log=True,                                       \
        ec="k",                                         \
        align="center")

    plt.xscale("log")
    plt.yscale("linear")
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')
    plt.xlabel("f")
    plt.ylabel("dBFS")
    plt.show()


def _display_filt(sos, fs, name='Filter'):
    """
    Display a sos-matrix over frequency and amplitude.

    Params
    ------
    `sos`:  sos-matrix of filter.
    `fs`:   Sampling rate of original signal.
    `name`: Name of the figure to be displayed

    Returns
    -------
    `None` (display function)
    """
    h, f = signal.sosfreqz(sos, worN=2048, fs=fs)

    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')
    plt.xlabel('[Hz]')
    plt.ylabel('[dB]')
    plt.title(name)
    plt.semilogx(h[1:], np.array([20 * np.log10(abs(value)) for value in f[1:]]), label=None)
    plt.ylim(-120, 20)        