import numpy as np
from numba import jit, prange, config
import multiprocessing
from tqdm import tqdm

@jit(nopython=True)
def scan(seismograms, timelags, saferest, sr, win, step):
    # saferest=max(timelags)
    t = np.arange(win / 2, len(seismograms[0]) / sr - saferest - win, step)
    br = np.zeros(t.shape)
    for index, i in enumerate(t):
        flag = timelags + i
        stack = np.zeros(round(win * sr))
        n = -1
        for j in range(seismograms.shape[0]):
            n = n + 1
            stack = stack + seismograms[j][
                            int(round((flag[n] - win / 4) * sr)):int(round((flag[n] - win / 4) * sr + win * sr))]
        br[index] = (stack ** 2).sum().item()

    return (br)


@jit(nopython=True)
def scanmax(seismograms, timelags, saferest, sr, win, step):
    # saferest=max(timelags)
    t = np.arange(win / 2, len(seismograms[0]) / sr - saferest - win, step)
    br = np.zeros(t.shape)
    for index, i in enumerate(t):
        flag = timelags + i
        stack = 0
        n = -1
        for j in range(seismograms.shape[0]):
            n = n + 1
            #stack = stack + seismograms[j][int(round(flag[n] * sr))]
            stack = stack + np.max(seismograms[j][int(round((flag[n] - step / 2) * sr)):int(round((flag[n] - step / 2) * sr + step * sr))])
        br[index] = stack

    return br


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def kur_scan(timelags, vn, sr, win, step):
    saferest = np.max(timelags)
    brshape = scanmax(seismograms=vn, timelags=timelags[0], saferest=saferest, sr=sr, win=win, step=step).shape
    br = np.zeros((timelags.shape[0], brshape[0]))
    for i in range(timelags.shape[0]):
        br[i] = scanmax(seismograms=vn, timelags=timelags[i], saferest=saferest, sr=sr, win=win, step=step)
        if i % 100 == 0:
            print(i)
    return br


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def med_scan(timelags, vn, sr, win, step):
    saferest = np.max(timelags)
    brshape = scan(seismograms=vn, timelags=timelags[0], saferest=saferest, sr=sr, win=win, step=step).shape
    br = np.zeros((timelags.shape[0], brshape[0]))
    for i in range(timelags.shape[0]):
        br[i] = scan(seismograms=vn, timelags=timelags[i], saferest=saferest, sr=sr, win=win, step=step)
        if i % 100 == 0:
            print(i)
    return br



@jit(nopython=True, parallel=True)
def med_scan_parallel(timelags, vn, sr, win, step):
    print("Number of threads used by Numba:", config.NUMBA_NUM_THREADS)
    saferest = np.max(timelags)
    brshape = scan(seismograms=vn, timelags=timelags[0], saferest=saferest, sr=sr, win=win, step=step).shape
    br = np.zeros((timelags.shape[0], brshape[0]))
    for i in prange(timelags.shape[0]):
        br[i] = scan(seismograms=vn, timelags=timelags[i], saferest=saferest, sr=sr, win=win, step=step)
        if i % 100 == 0:
            print(i)
    return br


#@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def regional_scan(timelags, vn, sr, win, step):
    saferest = np.max(timelags)
    brshape = scan(seismograms=vn, timelags=timelags[0], saferest=saferest, sr=sr, win=win, step=step).shape
    br = np.zeros((timelags.shape[0], brshape[0]))
    inputs = []
    for i in range(timelags.shape[0]):
        sortdis = np.argsort(timelags[i])
        vscan = []
        timelagscan = []
        for j in range(20):
            vscan.append(vn[sortdis[j]])
            timelagscan.append(timelags[i][sortdis[j]])

        vscan = np.array(vscan)
        timelagscan = np.array(timelagscan)

        br[i] = scan(seismograms=vscan, timelags=timelagscan, saferest=saferest, sr=sr, win=win, step=step)

    return br


@jit(nopython=True)
def stacking(t, timelags, win, sr, seismograms, br):
    for index, i in enumerate(t):
        flag = timelags + i
        stack = np.zeros(round(win * sr))
        n = -1
        for j in range(seismograms.shape[0]):
            n = n + 1
            stack = stack + seismograms[j][int(round((flag[n] - win / 4) * sr)):int(round((flag[n] - win / 4) * sr + win * sr))]
        br[index] = (stack ** 2).sum().item()

    return br


def scan_mul(input):
    seismograms = input[0]
    timelags = input[1]
    saferest = input[2]
    sr = input[3]
    win = input[4]
    step = input[5]
    if input[6] % 100 == 0:
        print(input[6])
    t = np.arange(win / 2, len(seismograms[0]) / sr - saferest - win, step)
    br = np.zeros(t.shape)
    br = stacking(t, timelags, win, sr, seismograms, br)

    return br


def med_scan_mul(timelags, vn, sr, win, step, processes):
    saferest = np.max(timelags)
    inputs = []
    for i in range(timelags.shape[0]):
        inputs.append([vn, timelags[i], saferest, sr, win, step, i])

    with multiprocessing.Pool(processes) as pool:
        pool_outputs = pool.map(scan_mul, inputs)

    br = np.array(pool_outputs)
    return br

def regional_scan_mul(input):
    seismograms = input[0]
    timelags = input[1]
    saferest = input[2]
    sr = input[3]
    win = input[4]
    step = input[5]
    regional = input[7]

    sortdis = np.argsort(timelags)
    vscan = []
    timelagscan = []
    for j in range(regional):
        vscan.append(seismograms[sortdis[j]])
        timelagscan.append(timelags[sortdis[j]])

    vscan = np.array(vscan)
    timelagscan = np.array(timelagscan)

    if input[6] % 100 == 0:
        print(input[6])
    t = np.arange(win / 2, len(seismograms[0]) / sr - saferest - win, step)
    br = np.zeros(t.shape)
    br = stacking(t, timelagscan, win, sr, vscan, br)

    return br

def regional_mul(timelags, vn, sr, win, step, processes, regional):
    saferest = np.max(timelags)
    inputs = []
    for i in range(timelags.shape[0]):
        inputs.append([vn, timelags[i], saferest, sr, win, step, i, regional])

    with multiprocessing.Pool(processes) as pool:
        pool_outputs = pool.map(regional_scan_mul, inputs)

    br = np.array(pool_outputs)
    return br


def softdecay_scan_mul(input):
    seismograms = input[0]
    timelags = input[1]
    saferest = input[2]
    sr = input[3]
    win = input[4]
    step = input[5]
    sigma = input[7]

    if input[6] % 100 == 0:
        print(input[6])

    weights = np.exp(-timelags ** 2 / (2 * sigma ** 2))
    weights /= np.sum(weights)
    vscan = seismograms * weights[:, np.newaxis]

    #vscan = []
    #weights = np.zeros(len(seismograms))
    #for i in range(len(seismograms)):
    #    weights[i] = np.exp(-timelags[i]**2 / (2 * sigma**2))

    #for i in range(len(seismograms)):
    #    vscan.append(seismograms[i] * weights[i])
    #vscan = np.array(vscan)

    t = np.arange(win / 2, len(seismograms[0]) / sr - saferest - win, step)
    #br = np.zeros(t.shape)
    br = stacking(t, timelags, win, sr, vscan, np.zeros(t.shape))
    #br = br / totalweight

    return br

def softdecay_mul(timelags, vn, sr, win, step, processes, sigma):
    saferest = np.max(timelags)
    #inputs = []
    #for i in range(timelags.shape[0]):
    #    inputs.append([vn, timelags[i], saferest, sr, win, step, i, sigma])

    inputs = ((vn, timelags[i], saferest, sr, win, step, i, sigma) for i in range(timelags.shape[0]))

    with multiprocessing.Pool(processes) as pool:
        pool_outputs = pool.map(softdecay_scan_mul, inputs)

    br = np.array(pool_outputs)
    return br


