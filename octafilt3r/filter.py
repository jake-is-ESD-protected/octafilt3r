import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def _gen_fc_fl_fu(_fmax, _fmin, _ratio):
    
    n_octs = m.log2(_fmax/_fmin)
    _n_bands = m.ceil(n_octs/_ratio) + 1
    _fcs = np.array([_fmin * 2. ** (i * _ratio) for i in range(0, _n_bands)])
    _fls = np.array([cur_fc * 2. ** ((-1) * _ratio / 2) for cur_fc in _fcs])
    _fus = np.array([cur_fc * 2. ** (_ratio / 2) for cur_fc in _fcs])

    return _fcs, _fls, _fus, _n_bands


def _gen_bandpass(_fs, _fl, _fu, _order=8, _display=False):
    ny = 1/2 * _fs
    lower = _fl / ny
    upper = _fu / ny
    # see https://dsp.stackexchange.com/questions/81285/sos-matrices-order-does-not-correspond-to-given-parameter-when-designing-bandpa
    _sos = signal.butter(int(_order/2), [lower, upper], btype='bandpass', output='sos') # obtain sos matrix

    if _display:
        h, f = signal.sosfreqz(_sos,worN=1024, fs=_fs)
        plt.semilogx(h[1:], np.array([20 * m.log10(abs(value)) for value in f[1:]]), label=None)
        plt.ylim(-120, 20)
        
    return _sos


def oct_bank(fs, fmax, fmin, ratio, order=8, display=False):

    fcs, fls, fus, n_bands = _gen_fc_fl_fu(fmax, fmin, ratio)
    sos = [[[]] for i in range(n_bands)]
    decimation_map = _get_dec_fct()

    for band in range(n_bands):
        sos[band] = _gen_bandpass(fs/decimation_map[band], fls[band], fus[band], _order=order, _display=display)

    return sos


def rolling_oct_bank(x, fs, ratio, order, fmax, fmin, window_size, n_decimations=4, dec_ord=10):

    # prepare signal
    x = _sig2list(x)
    n_frames = int(len(x)/window_size)
    fcs, fls, fus, n_bands = _gen_fc_fl_fu(fmax, fmin, ratio)

    # init arrays
    zis_banks = np.zeros((len(fcs), int(order/2), 2))
    zis_banks_next = zis_banks
    zis_dec = np.zeros((n_decimations, int(dec_ord/2), 2))
    zis_dec_next = zis_dec
    oct_features = np.zeros((n_frames, len(fcs)))
    y = np.zeros(window_size)


    # obtain filterbank
    sos = oct_bank(fs, fmax, fmin, ratio, order=order)
    sos_dec = decimator_filt(fs, order=dec_ord, display=False)
    

    for frame in range(n_frames):

        y = x[(frame * window_size):((frame + 1) * window_size)]
        spl = np.zeros([len(fcs)])
        act_band = len(fcs) - 1
        
        # no decimation
        for i in range(int(1/ratio) + 1):
            y_filt, zis_banks_next[act_band] = signal.sosfilt(sos[act_band], y, zi=zis_banks[act_band])
            spl[act_band] = 20 * np.log10((np.std(y_filt)) / 2e-5)
            act_band -= 1   

        y, zis_dec_next[0] = rolling_decimate(y, 2, sos_dec, zi=zis_dec[0])
        for i in range(int(1/ratio)):
            y_filt, zis_banks_next[act_band] = signal.sosfilt(sos[act_band], y, zi=zis_banks[act_band])
            spl[act_band] = 20 * np.log10((np.std(y_filt)) / 2e-5)
            act_band -= 1
            
        y, zis_dec_next[1] = rolling_decimate(y, 2, sos_dec, zi=zis_dec[1])
        for i in range(int(1/ratio)):
            y_filt, zis_banks_next[act_band] = signal.sosfilt(sos[act_band], y, zi=zis_banks[act_band])
            spl[act_band] = 20 * np.log10((np.std(y_filt)) / 2e-5)
            act_band -= 1
            
        y, zis_dec_next[2] = rolling_decimate(y, 2, sos_dec, zi=zis_dec[2])
        for i in range(int(1/ratio)):
            y_filt, zis_banks_next[act_band] = signal.sosfilt(sos[act_band], y, zi=zis_banks[act_band])
            spl[act_band] = 20 * np.log10((np.std(y_filt)) / 2e-5)
            act_band -= 1
            
        y, zis_dec_next[3] = rolling_decimate(y, 2, sos_dec, zi=zis_dec[3])
        remain = act_band + 1
        for i in range(remain):
            y_filt, zis_banks_next[act_band] = signal.sosfilt(sos[act_band], y, zi=zis_banks[act_band])
            spl[act_band] = 20 * np.log10((np.std(y_filt)) / 2e-5)
            act_band -= 1
             
        zis_banks = zis_banks_next
        oct_features[frame] = spl
        zis_dec = zis_dec_next

    return oct_features, fcs


def oct_spectrogram(features, fs, window_size):
    
    plt.pcolormesh(np.transpose(features), cmap = 'rainbow')
    plt.title('1/3 octave spectrogram')
    plt.xlabel("Frames")
    plt.ylabel("Bands")
    plt.colorbar()


def plot_bins(center_fqs, spl):
    
    spl_pad = np.zeros(len(spl))
    fig = plt.figure(figsize = (10, 5))

    for i in range(len(spl_pad)):

        start = min(spl) - 10
        stop = spl[i]

        if start < 0 and stop < 0:
            spl_pad[i] = abs(start - stop)
        if start < 0 and stop >= 0:
            spl_pad[i] = abs(start) + stop
        if start >= 0:
            spl_pad[i] = stop - start

    # padding for full display
    spl_pad = np.append(spl_pad, 0)
    fcs = np.append(center_fqs, center_fqs[-1] * 6/5)
    
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
    plt.ylabel("dB SPL")
    plt.show()

# be aware that this runs over in case of val not being between 0 and 1/-1
def _rms(buffer, root=True):
    b_sum = 0
    for val in buffer:
        b_sum = b_sum + val ** 2
    rms = b_sum / (len(buffer))

    if(root):
        rms = np.sqrt(rms)

    return rms


def _sig2list(x):
    if type(x) is list:
        return x
    elif type(x) is np.ndarray:
        return x.tolist()
    elif type(x) is tuple:
        return list(x)


def _get_dec_fct():
    factor = [  16, 16, 16, 16,     # custom decimation
                16, 16, 16, 16,
                16, 16, 16, 16,
                16, 16, 16,
                16, 16, 16,
                8, 8, 8,
                4, 4, 4,
                2, 2, 2,
                1, 1, 1, 1]
    return factor


def decimator_filt(fs, order=10, display=False):

    sos = signal.butter(
        N=order,
        Wn=(fs/5) / (fs/2),   # this results in the cutoff frequency being at fs/5
        btype='lowpass',
        analog=False,
        output='sos')

    if display:
        wn = 8192
        w = np.zeros(wn)
        h = np.zeros(wn, dtype=np.complex_)

        w[:], h[:] = signal.sosfreqz(
                sos,
                worN=wn,
                whole=False,
                fs=fs)
        
        fig, ax = plt.subplots()
        ax.semilogx(w, 20 * np.log10(abs(h) + np.finfo(float).eps), 'b')
        ax.grid(which='major')
        ax.grid(which='minor', linestyle=':')
        ax.set_xlabel(r'Frequency [Hz]')
        ax.set_ylabel('Amplitude [dB]')
        ax.set_title('Decimation filter')             

    return sos


# this is intended to be used in a loop
def rolling_decimate(x, factor, lp_sos, zi):

    sig = np.asarray(x)
    factor = int(factor)

    y, zi_new = signal.sosfilt(lp_sos, sig, zi=zi)

    for i in range(int(factor/2)):
        y = y[1::2]

    return y, zi_new
    