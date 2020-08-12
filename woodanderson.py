def wa(trace, delta):

    # from ground VEL (m/s) to woodanderson (mm)

    import numpy as np

    wa_gain = 2080
    wa_zeros = [0j]
    wa_poles = [(-6.2832 - 4.7124j), (-6.2832 + 4.7124j)]

    npts = len(trace.data)
    fs = 1 / delta
    f = np.arange(0, fs / 2, 1 / (npts * delta))
    s = 2 * np.pi * f * 1j

    wa_g = 1
    for i in wa_zeros:
        wa_g = wa_g * (s - i)

    for i in wa_poles:
        wa_g = wa_g / (s - i)

    wa_g = wa_gain * wa_g

    if npts % 2 > 0:
        wa_g_sub = wa_g[0:len(wa_g) - 1]
        wa_g_flip = wa_g_sub[::-1]
    else:
        wa_g_flip = wa_g[::-1]

    wa_g_complete = np.concatenate((wa_g, wa_g_flip))

    Y = np.fft.fft(trace.data)
    Y = Y * wa_g_complete * 1000     # m to mm
    trace.data = np.real(np.fft.ifft(Y))

    trace.filter('bandpass', freqmin=0.2, freqmax=20, zerophase=True)
    trace.detrend(type='demean')

    return (trace)