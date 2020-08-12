import pickle
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def mykurtosis(X):
    if len(X) == 0:
        return -3.0

    if np.var(X) == 0 or (np.var(X)**2) == 0:
        return -3.0

    m4 = 0
    mu = np.mean(X)
    for i in range(len(X)):
        m4 += (X[i] - mu)**4

    m4 = m4/len(X)
    return m4/(np.var(X)**2) - 3

def pick(points, dirsave, nametag, nsta,order, pp, ptraveltimes, straveltimes, step, win, kurwindow, sr, dstop, vfilter, h1filter, h2filter, vfilter2, h1filter2, h2filter2, savefigure, overlap):

    aspect = 8

    vertical = vfilter
    horizon1 = h1filter
    horizon2 = h2filter

    ponsets=np.ones([len(order), nsta]) * -1
    s1onsets=np.ones([len(order), nsta]) * -1
    s1kur=np.ones([len(order), nsta]) * -1
    s2onsets=np.ones([len(order), nsta]) * -1
    s2kur=np.ones([len(order), nsta]) * -1
    number=-1
    print("There are "+str(len(order))+"events.\n")

    pmarks=np.zeros([nsta, points])
    s1marks=np.zeros([nsta, points])
    s2marks=np.zeros([nsta, points])

    noise_count=0

    for i,nm in zip(order,pp):

        pick_countp=0
        pick_counts1=0
        pick_counts2=0
        taken_countp=0
        taken_counts1=0
        taken_counts2=0

        number=number+1
        print('event'+str(number))
        plt.clf()
        k=0
        onp=[]
        ons1=[]
        ons2=[]

        if i[1] * step <= overlap*60/2:
            continue

        if savefigure == 'on':
            fig = plt.figure(figsize=(11, 11))
            fff = fig.add_subplot(111)

        flagp=np.array(ptraveltimes[nm[0]])+i[1]*step+win/2    # flag is real time on the seismogram, referring the beginning of seismogram
        flags=np.array(straveltimes[nm[0]])+i[1]*step+win/2

        for j in vertical:
            k=k+1
            a=j[int(round((flagp[k-1]-9*win/4)*sr)):int(round((flagp[k-1]-win/4)*sr+3*win*sr))]
            if len(a)<1:
                a=np.ones(len(vertical[0][int(round((flagp[0]-9*win/4)*sr)):int(round((flagp[0]-win/4)*sr+3*win*sr))]))

            standard = len(a) / 2 - win / 4 * sr
            potential = range(int(standard - win / 4 * sr), int(standard + win * 3/4 * sr))

            kur = np.zeros(len(potential))
            numkur=0

            for potentialtime in potential:
                array=a[int(potentialtime-kurwindow*sr):potentialtime]
                kur[numkur]=mykurtosis(array)
                numkur=numkur+1

            kurrate=np.zeros(len(kur)-5)

            maxkr=0
            for ll in range(0,len(kur)-5):
                rate=kur[ll+5]-kur[ll]
                if rate>maxkr:
                    maxkr=rate
                    maxlocation=ll
                kurrate[ll]=rate

            kur_threshold=1.5
            kur_threshold2=1.5
            for ll in range(0,len(kurrate)):
                if kurrate[ll] > kur_threshold:
                    onsetflag=ll
                    break
                if ll == len(kurrate)-1:
                    if maxkr>kur_threshold2:
                        onsetflag=maxlocation-10  # -10 is from experience
                    else:
                        onsetflag=-1


            if onsetflag < 0:
                potential = range(int(standard - win/2 * sr-10), int(standard))
                kur = np.zeros(len(potential))

                numkur = 0
                for potentialtime in potential:
                    array = a[int(potentialtime - kurwindow * sr):potentialtime]
                    kur[numkur]=mykurtosis(array)
                    numkur=numkur+1

                maxkr = 0
                kurrate = np.zeros(len(kur) - 5)
                for ll in range(0, len(kur) - 5):
                    rate = kur[ll + 5] - kur[ll]
                    if rate > maxkr:
                        maxkr = rate
                        maxlocation = ll
                    kurrate[ll]=rate

                for ll in range(0, len(kurrate)):
                    if kurrate[ll] > kur_threshold:
                        onsetflag = ll
                        break
                    if ll == len(kurrate) - 1:
                        if maxkr > kur_threshold2:
                            onsetflag = maxlocation - 10  # -10 is from experience
                        else:
                            onsetflag = -1


            timedelay=(potential[onsetflag]-standard)/sr
            arrive=flagp[k-1]+timedelay

            if onsetflag<0:
                arrive=-1

            if savefigure == 'on':
                if onsetflag >= 0:
                    if pmarks[k-1][int(round(arrive * sr))] == 0:
                        plt.plot(a/max(abs(a))+2*k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='black', ms=5, markevery=[potential[onsetflag]])
                    else:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='red',ms=5, markevery=[potential[onsetflag]])
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
            #ponsets.append([-1] * nsta)
            #s1onsets.append([-1] * nsta)
            #s1kur.append([-1] * nsta)
            #s2onsets.append([-1] * nsta)
            #s2kur.append([-1] * nsta)
            continue

        ponsets[number] = onp

        if savefigure == 'on':
            fig = plt.figure(figsize=(11, 11))
            fff = fig.add_subplot(111)

        k=0
        kk=[]

        for j in horizon1:
            k = k + 1
            a = j[int(round((flags[k - 1] - 9 * win / 4) * sr)):int(round((flags[k - 1] - win / 4) * sr + 3 * win * sr))]
            if len(a)<1:
                a=np.ones(len(horizon1[0][int(round((flagp[0]-9*win/4)*sr)):int(round((flagp[0]-win/4)*sr+3*win*sr))]))

            standard = len(a) / 2 - win / 4 * sr
            potential = range(int(standard - win / 4 * sr), int(standard + 3 / 4 * win * sr))

            kur = np.zeros(len(potential))
            numkur = 0

            for potentialtime in potential:
                array = a[potentialtime - kurwindow * sr:potentialtime]
                kur[numkur] = mykurtosis(array)
                numkur = numkur + 1

            kurrate = np.zeros(len(kur) - 5)
            maxkr = 0
            for ll in range(0, len(kur) - 5):
                rate = kur[ll + 5] - kur[ll]
                if rate > maxkr:
                    maxkr = rate
                    maxlocation = ll
                kurrate[ll] = rate

            # kur_threshold = 4
            kur_threshold2 = 1.5

            if maxkr > kur_threshold2:
                onsetflag = maxlocation - 10  # -10 is from experience
            else:
                onsetflag = -1

            if onsetflag < 0:
                potential = range(int(standard - win / 2 * sr - 10), int(standard))
                kur = np.zeros(len(potential))
                numkur = 0
                for potentialtime in potential:
                    array = a[potentialtime - kurwindow * sr:potentialtime]
                    kur[numkur] = mykurtosis(array)
                    numkur = numkur + 1

                kurrate = np.zeros(len(kur) - 5)
                maxkr = 0
                for ll in range(0, len(kur) - 5):
                    rate = kur[ll + 5] - kur[ll]
                    if rate > maxkr:
                        maxkr = rate
                        maxlocation = ll
                    kurrate[ll] = rate

                if maxkr > kur_threshold2:
                    onsetflag = maxlocation - 10  # -10 is from experience
                else:
                    onsetflag = -1

            timedelay = (potential[onsetflag] - standard) / sr
            arrive = flags[k - 1] + timedelay

            if onsetflag < 0:
                arrive = -1

            if savefigure == 'on':
                if onsetflag >= 0:
                    if s1marks[k-1][int(round(arrive * sr))] == 0:
                        plt.plot(a/max(abs(a))+2*k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='black', ms=5, markevery=[potential[onsetflag]])
                    else:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='red',ms=5, markevery=[potential[onsetflag]])
                else:
                    plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, ls='-')

            if s1marks[k-1][int(round(arrive * sr))] == 1:
                taken_counts1=taken_counts1+1
                arrive=-1

            if arrive != -1:
                pick_counts1=pick_counts1+1
                for jy in range(int(round(flags[k-1]*sr)-win*sr/4*2), int(round(flags[k-1]*sr)+win*sr/4*3)):
                    s1marks[k-1][jy]=1

            ons1.append(arrive)
            kk.append(maxkr)

        if savefigure == 'on':
            fff.set_aspect(aspect)
            plt.savefig(dirsave + str(i[1]) + 's1_' +nametag + '.eps')
            plt.close('all')

        if taken_counts1 >= int(np.floor(pick_counts1) / 2):
            #ponsets.pop()
            ponsets[number] = [-1]*nsta
            #s1onsets.append([-1] * nsta)
            #s1kur.append(kk)
            #s2onsets.append([-1] * nsta)
            #s2kur.append([-1] * nsta)
            continue

        s1onsets[number] = ons1
        s1kur[number] = kk

        if savefigure == 'on':
            fig = plt.figure(figsize=(11, 11))
            fff = fig.add_subplot(111)

        k=0
        kk = []

        for j in horizon2:
            k = k + 1
            a = j[int(round((flags[k - 1] - 9 * win / 4) * sr)):int(round((flags[k - 1] - win / 4) * sr + 3 * win * sr))]
            if len(a)<1:
                a=np.ones(len(horizon2[0][int(round((flagp[0]-9*win/4)*sr)):int(round((flagp[0]-win/4)*sr+3*win*sr))]))

            standard = len(a) / 2 - win / 4 * sr
            potential = range(int(standard - win / 4 * sr), int(standard + win * 3 / 4 * sr))
            kur = np.zeros(len(potential))
            numkur = 0
            for potentialtime in potential:
                array = a[potentialtime - kurwindow * sr:potentialtime]
                kur[numkur] = mykurtosis(array)
                numkur = numkur + 1

            kurrate = np.zeros(len(kur) - 5)
            maxkr = 0
            for ll in range(0, len(kur) - 5):
                rate = kur[ll + 5] - kur[ll]
                if rate > maxkr:
                    maxkr = rate
                    maxlocation = ll
                kurrate[ll] = rate

            # kur_threshold = 4
            kur_threshold2 = 1.5

            if maxkr > kur_threshold2:
                onsetflag = maxlocation - 10  # -10 is from experience
            else:
                onsetflag = -1

            if onsetflag < 0:
                potential = range(int(standard - win / 2 * sr - 10), int(standard))
                kur = np.zeros(len(potential))
                numkur = 0
                for potentialtime in potential:
                    array = a[potentialtime - kurwindow * sr:potentialtime]
                    kur[numkur] = mykurtosis(array)
                    numkur = numkur + 1

                kurrate = np.zeros(len(kur) - 5)
                maxkr = 0
                for ll in range(0, len(kur) - 5):
                    rate = kur[ll + 5] - kur[ll]
                    if rate > maxkr:
                        maxkr = rate
                        maxlocation = ll
                    kurrate[ll] = rate

                if maxkr > kur_threshold2:
                    onsetflag = maxlocation - 10  # -10 is from experience
                else:
                    onsetflag = -1

            timedelay = (potential[onsetflag] - standard) / sr
            arrive = flags[k - 1] + timedelay

            if onsetflag < 0:
                arrive = -1

            if savefigure == 'on':
                if onsetflag >= 0:
                    if s2marks[k - 1][int(round(arrive * sr))] == 0:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='black',
                                 ms=5, markevery=[potential[onsetflag]])
                    else:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='red', ms=5,
                                 markevery=[potential[onsetflag]])
                else:
                    plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, ls='-')

            if s2marks[k - 1][int(round(arrive * sr))] == 1:
                taken_counts2=taken_counts2+1
                arrive = -1

            if arrive != -1:
                pick_counts2=pick_counts2+1
                for jy in range(int(round(flags[k - 1] * sr) - win * sr / 4 * 2),
                                int(round(flags[k - 1] * sr) + win * sr / 4 * 3)):
                    s2marks[k - 1][jy] = 1

            ons2.append(arrive)
            kk.append(maxkr)

        if savefigure == 'on':
            fff.set_aspect(aspect)
            plt.savefig(dirsave + str(i[1]) + 's2_' + nametag + '.eps')
            plt.close('all')

        if taken_counts2 >= int(np.floor(pick_counts2) / 2):
            #ponsets.pop()
            ponsets[number] = [-1]*nsta
            #s1onsets.pop()
            s1onsets[number] = [-1] * nsta
            #s2onsets.append([-1] * nsta)
            #s2kur.append(kk)
            continue

        s2onsets[number] = ons2
        s2kur[number] = kk

        if pick_countp + pick_counts1 + pick_counts2 < 10:
            noise_count=noise_count+1
        else:
            noise_count=0

        if noise_count==dstop:
            break

    ponsets=np.array(ponsets)
    #pickle.dump([ponsets],open(dirsave + 'ponsets_' + nametag + '.p','wb'))

    s1onsets=np.array(s1onsets)
    s1kur=np.array(s1kur)
    #pickle.dump([s1onsets],open(dirsave + 's1onsets_' + nametag + '.p','wb'))
    #pickle.dump([s1kur],open(dirsave + 's1kur_' + nametag + '.p','wb'))

    s2onsets=np.array(s2onsets)
    s2kur=np.array(s2kur)
    #pickle.dump([s2onsets],open(dirsave + 's2onsets_' + nametag + '.p','wb'))
    #pickle.dump([s2kur],open(dirsave + 's2kur_' + nametag + '.p','wb'))

    plt.close('all')


    #######################################################################
    # phase picking in another frequency band


    vertical=vfilter2
    horizon1 = h1filter2
    horizon2 = h2filter2

    ponsets2 = np.ones([len(order), nsta]) * -1
    s1onsets2 = np.ones([len(order), nsta]) * -1
    s1kur2 = np.ones([len(order), nsta]) * -1
    s2onsets2 = np.ones([len(order), nsta]) * -1
    s2kur2 = np.ones([len(order), nsta]) * -1
    number = -1
    print("There are " + str(len(order)) + "events.\n")

    pmarks = np.zeros([nsta, points])
    s1marks = np.zeros([nsta, points])
    s2marks = np.zeros([nsta, points])

    noise_count = 0

    for i, nm in zip(order, pp):

        pick_countp = 0
        pick_counts1 = 0
        pick_counts2 = 0
        taken_countp = 0
        taken_counts1 = 0
        taken_counts2 = 0

        number = number + 1
        print('event' + str(number))
        plt.clf()
        k = 0
        onp = []
        ons1 = []
        ons2 = []

        if i[1] * step <= overlap*60/2:
            continue

        if savefigure == 'on':
            fig = plt.figure(figsize=(11, 11))
            fff = fig.add_subplot(111)

        flagp = np.array(ptraveltimes[nm[0]]) + i[
                                                    1] * step + win / 2  # flag is real time on the seismogram, referring the beginning of seismogram
        flags = np.array(straveltimes[nm[0]]) + i[1] * step + win / 2

        for j in vertical:
            k = k + 1
            a = j[int(round((flagp[k - 1] - 9 * win / 4) * sr)):int(round((flagp[k - 1] - win / 4) * sr + 3 * win * sr))]
            if len(a)<1:
                a=np.ones(len(vertical[0][int(round((flagp[0]-9*win/4)*sr)):int(round((flagp[0]-win/4)*sr+3*win*sr))]))

            standard = len(a) / 2 - win / 4 * sr
            potential = range(int(standard - win / 4 * sr), int(standard + win * 3 / 4 * sr))

            kur = np.zeros(len(potential))
            numkur = 0

            for potentialtime in potential:
                array = a[int(potentialtime - kurwindow * sr):potentialtime]
                kur[numkur] = mykurtosis(array)
                numkur = numkur + 1

            kurrate = np.zeros(len(kur) - 5)

            maxkr = 0
            for ll in range(0, len(kur) - 5):
                rate = kur[ll + 5] - kur[ll]
                if rate > maxkr:
                    maxkr = rate
                    maxlocation = ll
                kurrate[ll] = rate

            kur_threshold = 1.5
            kur_threshold2 = 1.5
            for ll in range(0, len(kurrate)):
                if kurrate[ll] > kur_threshold:
                    onsetflag = ll
                    break
                if ll == len(kurrate) - 1:
                    if maxkr > kur_threshold2:
                        onsetflag = maxlocation - 10  # -10 is from experience
                    else:
                        onsetflag = -1

            if onsetflag < 0:
                potential = range(int(standard - win / 2 * sr - 10), int(standard))
                kur = np.zeros(len(potential))

                numkur = 0
                for potentialtime in potential:
                    array = a[int(potentialtime - kurwindow * sr):potentialtime]
                    kur[numkur] = mykurtosis(array)
                    numkur = numkur + 1

                maxkr = 0
                kurrate = np.zeros(len(kur) - 5)
                for ll in range(0, len(kur) - 5):
                    rate = kur[ll + 5] - kur[ll]
                    if rate > maxkr:
                        maxkr = rate
                        maxlocation = ll
                    kurrate[ll] = rate

                for ll in range(0, len(kurrate)):
                    if kurrate[ll] > kur_threshold:
                        onsetflag = ll
                        break
                    if ll == len(kurrate) - 1:
                        if maxkr > kur_threshold2:
                            onsetflag = maxlocation - 10  # -10 is from experience
                        else:
                            onsetflag = -1

            timedelay = (potential[onsetflag] - standard) / sr
            arrive = flagp[k - 1] + timedelay

            if onsetflag < 0:
                arrive = -1

            if savefigure == 'on':
                if onsetflag >= 0:
                    if pmarks[k - 1][int(round(arrive * sr))] == 0:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='black',
                                 ms=5, markevery=[potential[onsetflag]])
                    else:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='red', ms=5,
                                 markevery=[potential[onsetflag]])
                else:
                    plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, ls='-')

            if pmarks[k - 1][int(round(arrive * sr))] == 1:
                taken_countp = taken_countp + 1
                arrive = -1

            if arrive != -1:
                pick_countp = pick_countp + 1
                for jy in range(int(round(flagp[k - 1] * sr) - win * sr / 4 * 2),
                                int(round(flagp[k - 1] * sr) + win * sr / 4 * 3)):
                    pmarks[k - 1][jy] = 1
            onp.append(arrive)

        if savefigure == 'on':
            fff.set_aspect(aspect)
            plt.savefig(dirsave + str(i[1]) + 'p_low_' + nametag + '.eps')
            plt.close('all')

        if taken_countp >= int(np.floor(pick_countp) / 2):
            #ponsets2.append([-1] * nsta)
            #s1onsets2.append([-1] * nsta)
            #s1kur2.append([-1] * nsta)
            #s2onsets2.append([-1] * nsta)
            #s2kur2.append([-1] * nsta)
            continue

        ponsets2[number] = onp

        if savefigure == 'on':
            fig = plt.figure(figsize=(11, 11))
            fff = fig.add_subplot(111)

        k = 0
        kk = []

        for j in horizon1:
            k = k + 1
            a = j[int(round((flags[k - 1] - 9 * win / 4) * sr)):int(round((flags[k - 1] - win / 4) * sr + 3 * win * sr))]
            if len(a)<1:
                a=np.ones(len(horizon1[0][int(round((flagp[0]-9*win/4)*sr)):int(round((flagp[0]-win/4)*sr+3*win*sr))]))

            standard = len(a) / 2 - win / 4 * sr
            potential = range(int(standard - win / 4 * sr), int(standard + 3 / 4 * win * sr))

            kur = np.zeros(len(potential))
            numkur = 0

            for potentialtime in potential:
                array = a[potentialtime - kurwindow * sr:potentialtime]
                kur[numkur] = mykurtosis(array)
                numkur = numkur + 1

            kurrate = np.zeros(len(kur) - 5)
            maxkr = 0
            for ll in range(0, len(kur) - 5):
                rate = kur[ll + 5] - kur[ll]
                if rate > maxkr:
                    maxkr = rate
                    maxlocation = ll
                kurrate[ll] = rate

            # kur_threshold = 4
            kur_threshold2 = 1.5

            if maxkr > kur_threshold2:
                onsetflag = maxlocation - 10  # -10 is from experience
            else:
                onsetflag = -1

            if onsetflag < 0:
                potential = range(int(standard - win / 2 * sr - 10), int(standard))
                kur = np.zeros(len(potential))
                numkur = 0
                for potentialtime in potential:
                    array = a[potentialtime - kurwindow * sr:potentialtime]
                    kur[numkur] = mykurtosis(array)
                    numkur = numkur + 1

                kurrate = np.zeros(len(kur) - 5)
                maxkr = 0
                for ll in range(0, len(kur) - 5):
                    rate = kur[ll + 5] - kur[ll]
                    if rate > maxkr:
                        maxkr = rate
                        maxlocation = ll
                    kurrate[ll] = rate

                if maxkr > kur_threshold2:
                    onsetflag = maxlocation - 10  # -10 is from experience
                else:
                    onsetflag = -1

            timedelay = (potential[onsetflag] - standard) / sr
            arrive = flags[k - 1] + timedelay

            if onsetflag < 0:
                arrive = -1

            if savefigure == 'on':
                if onsetflag >= 0:
                    if s1marks[k - 1][int(round(arrive * sr))] == 0:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='black',
                                 ms=5, markevery=[potential[onsetflag]])
                    else:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='red', ms=5,
                                 markevery=[potential[onsetflag]])
                else:
                    plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, ls='-')

            if s1marks[k - 1][int(round(arrive * sr))] == 1:
                taken_counts1 = taken_counts1 + 1
                arrive = -1

            if arrive != -1:
                pick_counts1 = pick_counts1 + 1
                for jy in range(int(round(flags[k - 1] * sr) - win * sr / 4 * 2),
                                int(round(flags[k - 1] * sr) + win * sr / 4 * 3)):
                    s1marks[k - 1][jy] = 1

            ons1.append(arrive)
            kk.append(maxkr)

        if savefigure == 'on':
            fff.set_aspect(aspect)
            plt.savefig(dirsave + str(i[1]) + 's1_low_' + nametag + '.eps')
            plt.close('all')

        if taken_counts1 >= int(np.floor(pick_counts1) / 2):
            #ponsets2.pop()
            ponsets2[number] = [-1] * nsta
            #s1onsets2.append([-1] * nsta)
            #s1kur2.append(kk)
            #s2onsets2.append([-1] * nsta)
            #s2kur2.append([-1] * nsta)
            continue

        s1onsets2[number] = ons1
        s1kur2[number] = kk

        if savefigure == 'on':
            fig = plt.figure(figsize=(11, 11))
            fff = fig.add_subplot(111)

        k = 0
        kk = []

        for j in horizon2:
            k = k + 1
            a = j[int(round((flags[k - 1] - 9 * win / 4) * sr)):int(round((flags[k - 1] - win / 4) * sr + 3 * win * sr))]
            if len(a)<1:
                a=np.ones(len(horizon2[0][int(round((flagp[0]-9*win/4)*sr)):int(round((flagp[0]-win/4)*sr+3*win*sr))]))

            standard = len(a) / 2 - win / 4 * sr
            potential = range(int(standard - win / 4 * sr), int(standard + win * 3 / 4 * sr))
            kur = np.zeros(len(potential))
            numkur = 0
            for potentialtime in potential:
                array = a[potentialtime - kurwindow * sr:potentialtime]
                kur[numkur] = mykurtosis(array)
                numkur = numkur + 1

            kurrate = np.zeros(len(kur) - 5)
            maxkr = 0
            for ll in range(0, len(kur) - 5):
                rate = kur[ll + 5] - kur[ll]
                if rate > maxkr:
                    maxkr = rate
                    maxlocation = ll
                kurrate[ll] = rate

            # kur_threshold = 4
            kur_threshold2 = 1.5

            if maxkr > kur_threshold2:
                onsetflag = maxlocation - 10  # -10 is from experience
            else:
                onsetflag = -1

            if onsetflag < 0:
                potential = range(int(standard - win / 2 * sr - 10), int(standard))
                kur = np.zeros(len(potential))
                numkur = 0
                for potentialtime in potential:
                    array = a[potentialtime - kurwindow * sr:potentialtime]
                    kur[numkur] = mykurtosis(array)
                    numkur = numkur + 1

                kurrate = np.zeros(len(kur) - 5)
                maxkr = 0
                for ll in range(0, len(kur) - 5):
                    rate = kur[ll + 5] - kur[ll]
                    if rate > maxkr:
                        maxkr = rate
                        maxlocation = ll
                    kurrate[ll] = rate

                if maxkr > kur_threshold2:
                    onsetflag = maxlocation - 10  # -10 is from experience
                else:
                    onsetflag = -1

            '''
            for ll in range(0, len(kurrate)):
    
                if kurrate[ll] > kur_threshold:
                    onsetflag = ll
                    break
                if ll == len(kurrate) - 1:
                    if maxkr > kur_threshold2:
                        onsetflag = maxlocation - 10  # -10 is from experience
                    else:
                        onsetflag = -1
            '''

            timedelay = (potential[onsetflag] - standard) / sr
            arrive = flags[k - 1] + timedelay

            if onsetflag < 0:
                arrive = -1

            if savefigure == 'on':
                if onsetflag >= 0:
                    if s2marks[k - 1][int(round(arrive * sr))] == 0:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='black',
                                 ms=5, markevery=[potential[onsetflag]])
                    else:
                        plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='red', ms=5,
                                 markevery=[potential[onsetflag]])
                else:
                    plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, ls='-')

            if s2marks[k - 1][int(round(arrive * sr))] == 1:
                taken_counts2 = taken_counts2 + 1
                arrive = -1

            if arrive != -1:
                pick_counts2 = pick_counts2 + 1
                for jy in range(int(round(flags[k - 1] * sr) - win * sr / 4 * 2),
                                int(round(flags[k - 1] * sr) + win * sr / 4 * 3)):
                    s2marks[k - 1][jy] = 1

            ons2.append(arrive)
            kk.append(maxkr)

        if savefigure == 'on':
            fff.set_aspect(aspect)
            plt.savefig(dirsave + str(i[1]) + 's2_low_' + nametag + '.eps')
            plt.close('all')

        if taken_counts2 >= int(np.floor(pick_counts2) / 2):
            #ponsets2.pop()
            ponsets2[number] = [-1] * nsta
            #s1onsets2.pop()
            s1onsets2[number] = [-1] * nsta
            #s2onsets2.append([-1] * nsta)
            #s2kur2.append(kk)
            continue

        s2onsets2[number] = ons2
        s2kur2[number] = kk

        if pick_countp + pick_counts1 + pick_counts2 < 10:
            noise_count = noise_count + 1
        else:
            noise_count = 0

        if noise_count == dstop:
            break

    ponsets2=np.array(ponsets2)
    #pickle.dump([ponsets2],open(dirsave + 'ponsets2_' + nametag + '.p','wb'))

    s1onsets2=np.array(s1onsets2)
    s1kur2=np.array(s1kur2)
    #pickle.dump([s1onsets2],open(dirsave + 's1onsets2_' + nametag + '.p','wb'))
    #pickle.dump([s1kur2],open(dirsave + 's1kur2_' + nametag + '.p','wb'))

    s2onsets2=np.array(s2onsets2)
    s2kur2=np.array(s2kur2)
    #pickle.dump([s2onsets2],open(dirsave + 's2onsets2_' + nametag + '.p','wb'))
    #pickle.dump([s2kur2],open(dirsave + 's2kur2_' + nametag + '.p','wb'))

    ponsets_final = ponsets
    sonsets_final = s1onsets

    for i in range(0,len(ponsets)):
        for j in range(0,len(ponsets[i])):
            p1=ponsets[i][j]
            p2=ponsets2[i][j]
            if p1 > 0:
                ponsets_final[i][j]=p1
            elif p2 > 0:
                ponsets_final[i][j]=p2
            else:
                ponsets_final[i][j]=-1

            s1=s1onsets[i][j]
            kur1=s1kur[i][j]
            s2=s1onsets2[i][j]
            kur2=s1kur2[i][j]

            if s1>0 and s2>0:
                if abs(s1-s2) < 0.5:
                    tem1=(s1+s2)/2
                elif kur1>kur2:
                    tem1=s1
                else:
                    tem1=s2
            elif s1>0:
                tem1=s1
            elif s2>0:
                tem1=s2
            else:
                tem1=-1

            if tem1>0:
                sonsets_final[i][j]=tem1
            else:
                s3=s2onsets[i][j]
                kur3=s2kur[i][j]
                s4=s2onsets2[i][j]
                kur4=s2kur2[i][j]
                if s3>0 and s4>0:
                    if abs(s3-s4) < 0.5:
                        tem2=(s3+s4)/2
                    elif kur3>kur4:
                        tem2=s3
                    else:
                        tem2=s4
                elif s3>0:
                    tem2=s3
                elif s4>0:
                    tem2=s4
                else:
                    tem2=-1

                sonsets_final[i][j]=tem2

    return ([ponsets_final, sonsets_final])





