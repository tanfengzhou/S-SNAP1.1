import numpy as np
import distance
import taupz
from numba import jit
import getbeta

@jit(nopython=True)
def maxiscan(pstation, ps_station, ps_arrival, ptraveltimes, straveltimes, terr, maxi):
    for j in range(0, len(ps_station) - 1):
        for k in range(j + 1, len(ps_station)):
            obdiff = ps_arrival[j] - ps_arrival[k]
            for l in range(0, len(ptraveltimes)):
                if j < len(pstation):
                    t1 = ptraveltimes[l][ps_station[j]]
                else:
                    t1 = straveltimes[l][ps_station[j]]
                if k < len(pstation):
                    t2 = ptraveltimes[l][ps_station[k]]
                else:
                    t2 = straveltimes[l][ps_station[k]]
                thdiff = t1 - t2
                if abs(thdiff - obdiff) < terr * (1 / 3):
                    maxi[l] = maxi[l] + 5
                    continue
                if abs(thdiff - obdiff) < terr * 0.5:
                    maxi[l] = maxi[l] + 4
                    continue
                if abs(thdiff - obdiff) < terr:
                    maxi[l] = maxi[l] + 3
                    continue
                if abs(thdiff - obdiff) < terr * 2:
                    maxi[l] = maxi[l] + 2
                    continue
                if abs(thdiff - obdiff) < terr * 3:
                    maxi[l] = maxi[l] + 1
    return maxi


def MAXI_locate(ponsets, sonsets, ptraveltimes, straveltimes, stlas, stlos, stz, lowq, highq, studygrids, Q_threshold, terr, outlier, tableP, tableS, latgrid, longrid, depgrid, mindepgrid, minimprove):

    total=len(ponsets)
    print('There are '+str(total)+' events.\n')
    events=[]
    MAXI=[]
    catalog=[]

    for i in range(0,total):

        p=ponsets[i]
        s=sonsets[i]

        PandS = 0
        for j in range(0, len(p)):
            if p[j]>0 and s[j]>0:
                PandS = PandS + 1

        positivep=0
        parrival=[]
        pstation=[]
        num=0
        for j in p:
            if j>0:
                parrival.append(j)
                pstation.append(num)
                positivep = positivep + 1
            num=num+1

        positives = 0
        sarrival = []
        sstation = []
        num = 0
        for j in s:
            if j>0:
                sarrival.append(j)
                sstation.append(num)
                positives = positives + 1
            num=num+1

        if PandS < lowq:
            print('No.'+str(i)+' error: not enough picking. \n')
            events.append([-1,-1,-1,-1,-1,-1])
            MAXI.append(np.zeros(len(ptraveltimes)))
            continue


        maxi = np.zeros(len(ptraveltimes))
        full=positives+positivep
        MAXI_ceil=full*(full-1)/2
        ps_station=np.concatenate([pstation,sstation])
        ps_arrival=np.concatenate([parrival,sarrival])
        maxi=maxiscan(pstation, ps_station, ps_arrival, np.array(ptraveltimes), np.array(straveltimes), terr, maxi)
        m=max(maxi)
        Q=m/MAXI_ceil/5

        if PandS < highq:
            Q = 0

        print('No.' + str(i) + ' : Quality = ' + str(Q))

        p = [j for j, k in enumerate(maxi) if k == m]
        print(len(p))
        print(p[int(np.floor(len(p)/2))])
        for wei in p:
            print(studygrids[wei])
        evla = studygrids[p[int(np.floor(len(p)/2))]][0]
        evlo = studygrids[p[int(np.floor(len(p)/2))]][1]
        evdep = studygrids[p[int(np.floor(len(p)/2))]][2]
        p[0]=p[int(np.floor(len(p)/2))]

        estimatetimes=[]
        for j in range(0,len(pstation)):
            t=parrival[j]-ptraveltimes[p[0]][pstation[j]]
            estimatetimes.append(t)

        for j in range(0,len(sstation)):
            t=sarrival[j]-straveltimes[p[0]][sstation[j]]
            estimatetimes.append(t)

        eventtime0=np.median(estimatetimes)

        events.append([Q, m, eventtime0, evla, evlo, evdep])
        MAXI.append(maxi)

        if Q < Q_threshold:
            print('No.' + str(i) + ' low quality event \n')
            catalog.append([eventtime0, evla, evlo, evdep, -1, -1])
            print([eventtime0, evla, evlo, evdep, -1, -1])
            continue

        startrefinelatgrid = latgrid/2  # degree
        startrefinelongrid = longrid/2  # degree
        startrefinedepgrid = depgrid/2  # km
        lats = np.arange(evla-startrefinelatgrid*2, evla+startrefinelatgrid*2+startrefinelatgrid/10, startrefinelatgrid)
        lons = np.arange(evlo-startrefinelongrid*2, evlo+startrefinelongrid*2+startrefinelongrid/10, startrefinelongrid)
        deps = np.arange(evdep-startrefinedepgrid*2, evdep+startrefinedepgrid*2+startrefinedepgrid/10, startrefinedepgrid)

        for d in range(0, len(deps)):
            if deps[d] < 0:
                deps[d] = 0

        refinestudygrids = []
        for l in lats:
            for j in lons:
                for k in deps:
                    refinestudygrids.append([l, j, k])

        refinetraveldis = []
        for l in refinestudygrids:
            a = []
            for j, k in zip(stlas, stlos):
                a.append(distance.dis(j, k, l[0], l[1]))

            refinetraveldis.append(a)

        beta = getbeta.beta(evdep)
        timerangep = np.arange(0, startrefinedepgrid * 2 * 1.73 / beta + startrefinedepgrid * 2  * 1.73 / beta / 10, startrefinedepgrid / beta/2 )
        timerangen = -np.flip(timerangep,0)

        timerange = []
        for k in timerangen:
            timerange.append(k)
        timerange.pop()
        for k in timerangep:
            timerange.append(k)

        refineptraveltimes = []
        refinestraveltimes = []
        for l in range(0, len(refinetraveldis)):
            a = []
            for j in range(0,len(refinetraveldis[l])):
                time=taupz.taupz(tableP=tableP,tableS=tableS,dep=refinestudygrids[l][2], dis=refinetraveldis[l][j],phase='P',al=stz[j])
                a.append(time)

            refineptraveltimes.append(a)

            b = []
            for j in range(0,len(refinetraveldis[l])):
                time=taupz.taupz(tableP=tableP,tableS=tableS,dep=refinestudygrids[l][2], dis=refinetraveldis[l][j],phase='S',al=stz[j])
                b.append(time)

            refinestraveltimes.append(b)


        for j in range(0,len(pstation)):
            ot=parrival[j]-eventtime0
            tht=ptraveltimes[p[0]][pstation[j]]
            if abs(ot-tht) > outlier:
                pstation[j]=-1

        for j in range(0,len(sstation)):
            ot=sarrival[j]-eventtime0
            tht=straveltimes[p[0]][sstation[j]]
            if abs(ot-tht) > outlier:
                sstation[j]=-1

        remains=[]
        potimes=[]
        for j in range(0,len(refinestudygrids)):
            minremain = 9999
            n=-1

            for k in timerange:
                remain = 0
                potime=eventtime0+k
                n=n+1
                clean=0
                for l in range(0,len(pstation)):
                    if pstation[l] >= 0:
                        clean = clean + 1
                        remain=remain+(refineptraveltimes[j][pstation[l]]-(parrival[l]-potime))**2
                for l in range(0,len(sstation)):
                    if sstation[l] >= 0:
                        clean = clean + 1
                        remain=remain+(refinestraveltimes[j][sstation[l]]-(sarrival[l]-potime))**2

                remain=(remain/clean)**0.5
                if remain<minremain:
                    minremain=remain
                    minrenum=n

            remains.append(minremain)
            potimes.append(eventtime0+timerange[minrenum])

        minimum=min(remains)
        print('minimum = '+str(minimum)+'s')
        print('clean stations: ' + str(clean))
        p = [j for j, k in enumerate(remains) if k == minimum]
        finaltime=potimes[p[0]]
        finallat, finallon, finaldep = refinestudygrids[p[0]][0], refinestudygrids[p[0]][1], refinestudygrids[p[0]][2]

    ########################################################################################################################
        refinelatgrid = startrefinelatgrid
        refinelongrid = startrefinelongrid
        refinedepgrid = startrefinedepgrid

        while refinedepgrid > mindepgrid:

            evla, evlo, evdep, eventtime0 = finallat, finallon, finaldep, finaltime
            refinelatgrid = refinelatgrid/2  # degree
            refinelongrid = refinelongrid/2  # degree
            refinedepgrid = refinedepgrid/2  # km
            lats = np.arange(evla - refinelatgrid, evla + refinelatgrid+refinelatgrid/10, refinelatgrid)
            lons = np.arange(evlo - refinelongrid, evlo + refinelongrid+refinelongrid/10, refinelongrid)
            deps = np.arange(evdep - refinedepgrid, evdep + refinedepgrid+refinedepgrid/10, refinedepgrid)
            for d in range(0,len(deps)):
                if deps[d]<0:
                    deps[d]=0

            refinestudygrids = []
            for l in lats:
                for j in lons:
                    for k in deps:
                        refinestudygrids.append([l, j, k])

            refinetraveldis = []
            for l in refinestudygrids:
                a = []
                for j, k in zip(stlas, stlos):
                    a.append(distance.dis(j, k, l[0], l[1]))

                refinetraveldis.append(a)

            beta = getbeta.beta(evdep)
            timerangep = np.arange(0,refinedepgrid * 1.73 / beta + refinedepgrid * 1.73 / beta / 10, refinedepgrid / beta/2)
            timerangen = -np.flip(timerangep,0)

            timerange = []
            for k in timerangen:
                timerange.append(k)
            timerange.pop()
            for k in timerangep:
                timerange.append(k)

            refineptraveltimes = []
            refinestraveltimes = []
            for l in range(0, len(refinetraveldis)):
                a = []
                for j in range(0, len(refinetraveldis[l])):
                    time = taupz.taupz(tableP=tableP,tableS=tableS, dep=refinestudygrids[l][2], dis=refinetraveldis[l][j],phase='P', al=stz[j])
                    a.append(time)

                refineptraveltimes.append(a)

                b = []
                for j in range(0, len(refinetraveldis[l])):
                    time=taupz.taupz(tableP=tableP,tableS=tableS, dep=refinestudygrids[l][2], dis=refinetraveldis[l][j], phase='S', al=stz[j])
                    b.append(time)

                refinestraveltimes.append(b)

            remains = []
            potimes = []
            for j in range(0, len(refinestudygrids)):
                minremain = 9999
                n = -1

                for k in timerange:
                    remain=0
                    potime = eventtime0 + k
                    n = n + 1
                    clean = 0
                    for l in range(0, len(pstation)):
                        if pstation[l] >= 0:
                            clean = clean + 1
                            remain = remain + (refineptraveltimes[j][pstation[l]] - (parrival[l] - potime)) ** 2
                    for l in range(0, len(sstation)):
                        if sstation[l] >= 0:
                            clean = clean + 1
                            remain = remain + (refinestraveltimes[j][sstation[l]] - (sarrival[l] - potime)) ** 2

                    remain = (remain / clean) ** 0.5
                    if remain < minremain:
                        minremain = remain
                        minrenum = n

                remains.append(minremain)
                potimes.append(eventtime0 + timerange[minrenum])

            improve = (minimum-min(remains))/minimum
            minimum = min(remains)
            print('refinegrid = '+str(refinedepgrid)+'km')
            print('minimum = '+str(minimum)+'s')
            print('improve = '+str(improve*100)+"%")
            print('clean stations: ' + str(clean))
            p = [j for j, k in enumerate(remains) if k == minimum]
            finaltime = potimes[p[0]]
            finallat, finallon, finaldep = refinestudygrids[p[0]][0], refinestudygrids[p[0]][1], refinestudygrids[p[0]][2]
            finalgrid = refinedepgrid
            if improve < minimprove:
                refinelatgrid = startrefinelatgrid
                refinelongrid = startrefinelongrid
                refinedepgrid = startrefinedepgrid
                break

            elif refinedepgrid < mindepgrid:
                refinelatgrid = startrefinelatgrid
                refinelongrid = startrefinelongrid
                refinedepgrid = startrefinedepgrid
                break

    ########################################################################################################################

        catalog.append([finaltime, finallat, finallon, finaldep, minimum, finalgrid, clean])
        print([finaltime, finallat, finallon, finaldep, minimum, finalgrid, clean])

    #pickle.dump([events],open(dirsave + 'MAXI_PRED' + nametag +'.p','wb'))
    #pickle.dump([MAXI],open(dirsave + 'MAXIvalue' + nametag +'.p','wb'))
    #pickle.dump([catalog],open(dirsave + 'cataloghigh' + nametag +'.p','wb'))

    return [events, MAXI, catalog]











