# ------[Octafilt3r]------
#   @ name: octafilter.filter
#   @ auth: Jakob Tschavoll
#   @ vers: 0.2
#   @ date: 2022

"""
Module for octave-based filtering and spectrograms. Intended to be used for
feature extraction for convolutional neural networks in audio classification.
"""

import numpy as np
import scipy.signal as signal
from octafilt3r import plot as o3plot

__all__ = ['rolling_oct_bank', 'oct_bank']


def rolling_oct_bank(x, fs, ratio=1/3, order=8, fmax=20000, fmin=20, frame_size=2000, n_decimations=4, dec_ord=10):
    """
    Create a complete octave-filterbank of desired ratio, order and bandwidth which
    direcctly computes the dBFS value of each frame of data of `window_size`. Both
    the bandpass and decimation filters are butterworth filters.

    Params
    ------
    `x`:                Input signal to filter and analyze.
    `fs`:               Sampling rate of input signal.
    `ratio`:            Octave split.
    `order`:            Order of bandpassfilters (at least `order=6` is recommended).
    `fmax`:             Upper frequency limit of interest.
    `fmin`:             Lower frequency limit of interest.
    `frame_size`:       Size of step in samples. The whole signal is split accordingly
                        and analyzed in these steps. The smaller the size, the higher
                        the definition in time, but the more fuzzy the overall filter
                        quality.
    `n_decimations`:    Number of desired decimations. A decimation applies an `anti-aliasing`
                        filter and then halves the sample rate for faster computations. The
                        position of decimations is fixed according to this diagram:

                        `band[n], band[n-1], band[n-2], band[n-3]`

                        `--decimate 1--`

                        `band[n-4], band[n-5], band[n-6],`

                        `--decimate 2--`

                        `band[n-7], band[n-8], band[n-9],`

                        `--decimate 3--`

                        `...`

                        `--decimate n--`

                        `remaining bands`
                        
    `dec_ord`:          Order of decimation AA-filters.

    Returns
    -------
    `oct_features`:     feature matrix of dBFS levels across all frames present in `x`
                        of shape `(n_frames, n_bands)`
    `fcs`:              array of center-frequencies of all bands

    Notes
    -----
    The bank is generated best when `fmin` is a `log2` of `fmin`, since octaves correspond to a doubling
    in frequency.
    ## IMPORTANT:
    If the requested `fmax` has an upper cutoff-frequency (`-3dB point`) that is greater than `fs/2` (nyquist)
    then the whole band will be missing to avoid aliasing. Requirement: `fmax * 2**(ratio/2) < fs/2`
    """


    # prepare signal
    x = _sig2list(x)
    n_frames = int(len(x)/frame_size)
    fcs, fls, fus, n_bands = _gen_fc_fl_fu(fmax, fmin, ratio)
    pascal = 1 # dummy value, causes result to be in dBFS
    lim_zero = 1e-20

    # init arrays
    zis_banks = np.zeros((len(fcs), int(order/2), 2))
    zis_banks_next = zis_banks
    zis_dec = np.zeros((n_decimations, int(dec_ord/2), 2))
    zis_dec_next = zis_dec
    oct_features = np.zeros((n_frames, len(fcs)))
    y = np.zeros(frame_size)


    # obtain filterbank
    sos = oct_bank(fs, fmax, fmin, ratio, n_decimations, order=order)
    sos_dec = _decimator_filt(fs, order=dec_ord, display=False)
    

    for frame in range(n_frames):

        y = x[(frame * frame_size):((frame + 1) * frame_size)]
        spl = np.zeros([len(fcs)])
        act_band = len(fcs) - 1
        d = 0
        
        # no decimation
        for i in range(int(1/ratio) + 1):
            y_filt, zis_banks_next[act_band] = signal.sosfilt(sos[act_band], y, zi=zis_banks[act_band])
            spl[act_band] = 10 * np.log10((_rms(y_filt, False) + lim_zero) / pascal)
            act_band -= 1

        # desired decimations
        for d in range(n_decimations - 1):
            y, zis_dec_next[d] = _rolling_decimate(y, sos_dec, zi=zis_dec[d])
            for i in range(int(1/ratio)):
                y_filt, zis_banks_next[act_band] = signal.sosfilt(sos[act_band], y, zi=zis_banks[act_band])
                spl[act_band] = 10 * np.log10((_rms(y_filt, False) + lim_zero) / pascal)
                act_band -= 1

        # last decimation
        y, zis_dec_next[d+1] = _rolling_decimate(y, sos_dec, zi=zis_dec[d+1])
        remain = act_band + 1
        for i in range(remain):
            y_filt, zis_banks_next[act_band] = signal.sosfilt(sos[act_band], y, zi=zis_banks[act_band])
            spl[act_band] = 10 * np.log10((_rms(y_filt, False) + lim_zero) / pascal)
            act_band -= 1
             
        zis_banks = zis_banks_next
        oct_features[frame] = spl
        zis_dec = zis_dec_next

    return oct_features, fcs


def oct_bank(fs, fmax, fmin, ratio, n_decimations, order=8, display=False):
    """
    Create the filterbank itself via butterworth bandpasses.

    Params
    ------
    `fs`:               Sampling rate of later filtered signal.
    `fmax`:             Upper frequency limit of interest.
    `fmin`:             Lower frequency limit of interest.
    `ratio`:            Octave split.
    `n_deciamtions`:    Number of decimations across filterbank
    `order`:            Order of bandpassfilters (at least `order=6` is recommended).
    `display`:          Boolean for visual display of the whole filterbank

    Returns
    -------
    `sos_bm`:       sos-matrix array of coefficients  for each band of shape `(n_bands, order/2, 6)`
    
    """
    fcs, fls, fus, n_bands = _gen_fc_fl_fu(fmax, fmin, ratio)
    sos_bm = [[[]] for i in range(n_bands)]
    decimation_map = _get_dec_fct(n_bands, n_decimations)

    for band in range(n_bands):
        sos_bm[band] = _gen_bandpass(fs/decimation_map[band], fls[band], fus[band], order=order, display=display)

    return sos_bm


def _rms(x, root=True):
    """
    Classic implementation of `RMS`-calculation.

    Params
    ------
    `x`:    Input signal.
    `root`: Boolean for need for `sqrt()`-operation. (Not needed in power-calculations)

    Returns
    -------
    `rms`:  Skalar rms-value of `x`
    """

    b_sum = 0
    for val in x:
        b_sum += (val ** 2)
    rms = b_sum / (len(x))

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


def _get_dec_fct(n_bands=31, n_decimations=4):
    """
    Obtain an array of decimation factors specific for number of bands and decimations.

    Params
    ------
    `n_bands`:          Number of bands in filterbank.
    `n_decimations`:    Number of desired decimations.
                        The position of decimations is fixed according to this diagram:

                        `band[n], band[n-1], band[n-2], band[n-3]`

                        `--decimate 1--`

                        `band[n-4], band[n-5], band[n-6],`

                        `--decimate 2--`

                        `band[n-7], band[n-8], band[n-9],`

                        `--decimate 3--`

                        `...`

                        `--decimate n--`

                        `remaining bands`

    Returns
    -------
    `factor`:           Array of decimation factors starting with the highest factor (lowest band)

    Notes
    -----
    Returns `None` and displays an error in case of too many decimations requested for too few bands.

    """
    act_band = 4
    factor = [1,1,1,1]
    exp = 1
    
    for i in range(n_decimations):

        for j in range(3):
            factor.append(2**exp)
            act_band += 1
        exp += 1    

    remain = n_bands - act_band
    if remain < 0:
        print(f'DECIMATION ERROR in _get_dec_fct(n_bands={n_bands}, n_decimations={n_decimations}):')
        print(f'\n<{n_bands}> bands are not enough to be decimated <{n_decimations}> times!\n')
        return None

    for i in range(remain):
        factor.append(2**(exp-1))

    return factor[::-1]


def _decimator_filt(fs, order=10, display=False):
    """
    Obtain the filter coefficients for the AA-lowpass (Butterworth) associated with the decimations.

    Params
    ------
    `fs`:       Sampling rate of original signal.
    `order`:    Order of the lowpass filter.
    `display`:  Boolean for displaying the created filter.

    Returns
    -------
    `sos`:      sos-matrix for the filter.

    Notes
    -----
    Since the decimation reduces the sample rate, the filter coefficients stay the same for every decimation
    because they are calculated relative to the nyquist frequency `fs/2`. To obtain decimation that neither
    lets too much aliasing nor influence on the curent upper most band happen, the cutoff frequency of the
    AA filter is set to a constant `1/5.5` of the sampling rate
    """
    sos = signal.butter(
        N=order,
        Wn=(fs/5.5) / (fs/2),   # this results in the cutoff frequency being at fs/5.5
        btype='lowpass',
        analog=False,
        output='sos')

    if display:
        o3plot._display_filt(sos, fs, 'Decimation AA filter')

    return sos


def _rolling_decimate(x, lp_sos, zi):
    """
    Decimation function which applies the filter generated in `_decimator_filt`. To make continuous operation possible,
    initial conditions can be set and received from this function.

    Params
    ------
    `x`:        Signal to decimate.
    `lp_sos`:   sos-matrix of AA-filter
    `zi`:       Initial conditions for filter of shape `(order/2, 2)`

    Returns
    -------
    `y`:        Downsampled signal
    `zi_new`:   Current conditions which should then be passed to the next call of `_rolling_decimate`
                (same shape as `zi`)
    """
    sig = np.asarray(x)

    y, zi_new = signal.sosfilt(lp_sos, sig, zi=zi)

    return y[1::2], zi_new


def _gen_fc_fl_fu(fmax, fmin, ratio, IEC=True):
    """
    Generate the center, lower cutoff (-3dB point) and upper cutoff of all requested bands
    which represent an octave filterbank of a given ratio.

    Params
    ------
    `fmax`:     Upper frequency limit of interest.
    `fmin`:     Lower frequency limit of interest.
    `ratio`:    Octave split.
    `IEC`:      Boolean for auto-generation of frequencies or standardized by IEC61260

    Returns
    -------
    `fcs`:      Array of all center frequencies (`0dB` point of filter)
    `fls`:      Array of all lower cutoff frequencies (`-3dB` point of filter)
    `fus`:      Array of all upper cutoff frequencies (`-3dB` point of filter)
    `n_bands`:  Number of calculated bands (same as `len(fcs)`)
    """

    G = 10. ** (3. / 10.)

    nom_mid_frqs = np.array([
        25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 
        250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 
        2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 
        12500.0, 16000.0, 20000.0])

    select_f = []
    select_f = np.asarray(select_f)

    for frq in nom_mid_frqs:
        if ((frq <= fmax) and (frq >= fmin)):
            select_f = np.append(select_f, frq)
    
    if(IEC):
        fcs = select_f
        n_bands = len(fcs)

    else:
        n_bands = int(np.ceil((np.log2(fmax/fmin))/ratio))

        fcs = np.array([fmin * G ** (i * ratio) for i in range(0, n_bands)])

    fls = np.array([cur_fc * G ** ((-1) * ratio / 2) for cur_fc in fcs])
    fus = np.array([cur_fc * G ** (ratio / 2) for cur_fc in fcs])

    return fcs, fls, fus, n_bands


def _gen_bandpass(fs, fl, fu, order=8, display=False):
    """
    Generate coefficients for a single bandpass which satisfies octave filter requirements.

    Params
    ------
    `fs`:       Sampling rate of intended input signal.
    `fl`:       Lower cutoff frequency (`-3dB`-point).
    `fu`:       Upper cutoff frequency (`-3dB`-point).
    `order`:    Order of the filter.
    `display`:  Boolean for display of the filter.

    Returns
    -------
    `sos`:      sos-matrix of filter coefficients specific to that band

    Notes
    -----
    The general understanding of `order` for bandpass filters used to be ambiguous when passed to the
    `scipy`-function `signal.butter()`. See discussion [here](https://dsp.stackexchange.com/questions/81285/sos-matrices-order-does-not-correspond-to-given-parameter-when-designing-bandpa).
    This has since been updated in the documentation.
    """
    ny = 1/2 * fs
    l = fl / ny
    u = fu / ny
    sos = signal.butter(int(order/2), [l, u], btype='bandpass', output='sos') # obtain sos matrix

    if display:
        o3plot._display_filt(sos, fs, 'Filterbank')
        
    return sos