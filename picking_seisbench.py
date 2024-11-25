import copy

import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream
from scipy.signal import find_peaks

def pick(points, dirsave, nametag, nsta,order, pp, ptraveltimes, straveltimes, step, win, sr_original, dstop, v, h1, h2, savefigure, overlap, trained_model, aipick_threshold):

    aspect = 30
    sr = 100
    pscores = []
    ptimes = []
    sscores = []
    stimes = []

    for i,j,k in zip(v, h1, h2):
        st = Stream(i) + Stream(j) + Stream(k)
        st.resample(sampling_rate=sr)
        st.detrend("linear")
        phase_predictions = trained_model.annotate(st)
        print(phase_predictions)
        ppred = 0
        spred = 0
        for tr in phase_predictions:
            if tr.id.endswith('_P'):
                shijiancha = tr.stats.starttime - i.stats.starttime
                pscore = tr.data
                #plt.plot(pscore)
                ppeak, pproperties = find_peaks(pscore, height=aipick_threshold, distance=sr)
                ptimes.append(ppeak)
                pscores.append(pproperties['peak_heights'])
                ppred = 1
            if tr.id.endswith('_S'):
                sscore = tr.data
                speak, sproperties = find_peaks(sscore, height=aipick_threshold, distance=sr)
                stimes.append(speak)
                sscores.append(sproperties['peak_heights'])
                spred = 1

        if ppred == 0:
            ptimes.append([])
            pscores.append([])

        if spred == 0:
            stimes.append([])
            sscores.append([])


    ponsets=np.ones([len(order), nsta]) * -1
    sonsets=np.ones([len(order), nsta]) * -1

    number=-1
    print("There are "+str(len(order))+"events.\n")

    pmarks=np.zeros([nsta, int(round(points * sr / sr_original))])
    smarks=np.zeros([nsta, int(round(points * sr / sr_original))])

    noise_count=0

    for i,nm in zip(order,pp):

        pick_countp=0
        pick_counts=0
        taken_countp=0
        taken_counts=0

        number=number+1
        print('event'+str(number))
        plt.clf()
        k=0
        onp=[]
        ons=[]

        if i[1] * step <= overlap*60/2:
            continue

        if savefigure == 'on':
            fig = plt.figure(figsize=(11, 11))
            fff = fig.add_subplot(111)

        flagp=np.array(ptraveltimes[nm[0]])+i[1]*step+win/2    # flag is real time on the seismogram, referring the beginning of seismogram
        flags=np.array(straveltimes[nm[0]])+i[1]*step+win/2

        for j in v:

            k=k+1
            a=j.data[int(round((flagp[k-1]-9*win/4)*sr)):int(round((flagp[k-1]-win/4)*sr+3*win*sr))]

            if len(a)<1:
                a=np.ones(len(v[0].data[int(round((flagp[0]-9*win/4)*sr)):int(round((flagp[0]-win/4)*sr+3*win*sr))]))

            standard = len(a) / 2 - win / 4 * sr
            potential = range(int((flagp[k-1] - win / 2) * sr), int((flagp[k-1] + win / 2) * sr))

            arrive = -1
            ppp = []
            for pickpoint in ptimes[k-1]:
                picktime = pickpoint + int(round(shijiancha*sr))
                if picktime in potential:
                    ppp.append(picktime/sr)

            if len(ppp) == 0:
                arrive = -1
            else:
                arrive = min(ppp, key=lambda x: abs(x - flagp[k-1]))

            timedelay = arrive - flagp[k - 1]
            plotpick = int(round(timedelay * sr + standard))

            if savefigure == 'on':
                if arrive >= 0:
                    if pmarks[k-1][int(round(arrive * sr))] == 0:
                        plt.plot(a/max(abs(a))+2*k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='black', ms=5, markevery=[plotpick])
                    else:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='red',ms=5, markevery=[plotpick])
                else:
                    plt.plot(a/ max(abs(a)) + 2 * k, linewidth=0.7, ls='-')

            if pmarks[k-1][int(round(arrive * sr))] == 1:
                taken_countp=taken_countp+1
                arrive=-1

            if arrive != -1:
                pick_countp=pick_countp+1
                for jy in range(int(round(flagp[k-1]*sr)-win*sr/4*2), int(round(flagp[k-1]*sr)+win*sr/4*3)):
                    pmarks[k-1][jy]=1
            onp.append(arrive)

        if savefigure == 'on':
            fff.set_aspect(aspect)
            plt.savefig(dirsave+str(i[1])+'p_' + nametag + '.eps')
            plt.close('all')

        if taken_countp >= int(np.floor(pick_countp) / 2):
            continue

        ponsets[number] = onp

        if savefigure == 'on':
            fig = plt.figure(figsize=(11, 11))
            fff = fig.add_subplot(111)

        k=0
        for j in h1:
            k = k + 1
            a = j.data[int(round((flags[k - 1] - 9 * win / 4) * sr)):int(round((flags[k - 1] - win / 4) * sr + 3 * win * sr))]
            if len(a)<1:
                a=np.ones(len(h1[0].data[int(round((flagp[0]-9*win/4)*sr)):int(round((flagp[0]-win/4)*sr+3*win*sr))]))

            standard = len(a) / 2 - win / 4 * sr
            potential = range(int((flags[k-1] - win / 2) * sr), int((flags[k-1] + win / 2) * sr))

            arrive = -1
            sss = []
            for pickpoint in stimes[k - 1]:
                picktime = pickpoint + int(round(shijiancha*sr))
                if picktime in potential:
                    sss.append(picktime / sr)

            if len(sss) == 0:
                arrive = -1
            else:
                arrive = min(sss, key=lambda x: abs(x - flags[k - 1]))

            timedelay = arrive - flags[k - 1]
            plotpick = int(round(timedelay * sr + standard))

            if savefigure == 'on':
                if arrive >= 0:
                    if smarks[k-1][int(round(arrive * sr))] == 0:
                        plt.plot(a/max(abs(a))+2*k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='black', ms=5, markevery=[plotpick])
                    else:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='red',ms=5, markevery=[plotpick])
                else:
                    plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, ls='-')

            if smarks[k-1][int(round(arrive * sr))] == 1:
                taken_counts = taken_counts+1
                arrive=-1

            if arrive != -1:
                pick_counts = pick_counts+1
                for jy in range(int(round(flags[k-1]*sr)-win*sr/4*2), int(round(flags[k-1]*sr)+win*sr/4*3)):
                    smarks[k-1][jy]=1

            ons.append(arrive)

        if savefigure == 'on':
            fff.set_aspect(aspect)
            plt.savefig(dirsave + str(i[1]) + 's1_' +nametag + '.eps')
            plt.close('all')

        if taken_counts >= int(np.floor(pick_counts) / 2):
            ponsets[number] = [-1]*nsta
            continue

        sonsets[number] = ons

        if savefigure == 'on':
            fig = plt.figure(figsize=(11, 11))
            fff = fig.add_subplot(111)


        if pick_countp + pick_counts < 10:
            noise_count=noise_count+1
        else:
            noise_count=0

        if noise_count==dstop:
            break

    ponsets=np.array(ponsets)
    sonsets=np.array(sonsets)

    plt.close('all')

    ponsets_final = ponsets
    sonsets_final = sonsets

    return ([ponsets_final, sonsets_final])





