import numpy as np
import pickle
import taupz
import csv
import geodis
import obspy
from obspy import read
from pathlib import Path
import distance
import math
from psseparation import psseparation
import copy
import ssa
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks
from operator import itemgetter
import picking_seisbench
import locatePS
#from mpl_toolkits.basemap import Basemap
import calc_mag
import stations_plot
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

######################################################################

def main():
    nametag='allst'

    area='ca_fewer'
    studyarea=[35.3,-118.1,1,1]
    studydepth=[10,10]
    xygrid=1         #km
    depgrid=1        #km
    latgrid=xygrid/geodis.geodis([[studyarea[0]-0.5,studyarea[1]],[studyarea[0]+0.5,studyarea[1]]])     #degree
    longrid=xygrid/geodis.geodis([[studyarea[0],studyarea[1]-0.5],[studyarea[0],studyarea[1]+0.5]])     #degree

    studydepth_for_locate = [0, 20]

    time_begin=dict([
        ('year','2019'),
        ('month','07'),
        ('day','06'),
        ('hour','06'),
        ('min','00'),
        ('second', '00')
    ])

    time_end=dict([
        ('year','2019'),
        ('month','07'),
        ('day','06'),
        ('hour','07'),
        ('min','00'),
        ('second', '00')
    ])

    process_unit = 60       # min
    overlap = 5             # min

    datadir = './data2019/'
    outputdir = './07eqt/'
    xmldir = './dataless/'

    ######################################################################

    sr=40      # Hz
    scanlowf=5
    scanhighf=20
    root=3
    win=6       # seconds
    step=1      # seconds
    points=int(sr*(process_unit+overlap)*60)

    psseparate = 'off'
    brscan = 'on'            # brscan and kurscan cannot be both off
    kurscan = 'off'

    processes = 4

    ######################################################################

    #kurwindow=4     # second
    #picklf1=5       # Hz
    #pickhf1=20       # Hz     lower band
    #picklf2=5       # Hz
    #pickhf2=20      # Hz     higher band

    dstop=10        # have set to 100 when running in PGC or UVic cluster, should not be even any change

    savefigure = 'on'

    import seisbench.models as sbm
    trained_model = sbm.EQTransformer.from_pretrained("original")         # EQ Transformer (Mousavi et al., 2020, Nat Commun)
    #trained_model = sbm.EQTransformer.from_pretrained("scedc")
    #trained_model = sbm.PhaseNet.from_pretrained("obs")                  # PickBlue (Bornstein et al., 2023, Earth and Space Science)
    #trained_model = sbm.PhaseNet.from_pretrained("scedc")
    # other models available in Seisbench, see their instructions
    aipick_threshold = 0.05
    ######################################################################

    highq = 10
    lowq = 7
    terr = 1          # acceptable error in second
    Q_threshold = 0.4
    outlier = 1.5          # second
    mindepgrid = 0.1     # km
    minimprove = 0.001   # percent

    ######################################################################

    with open('./tableP_hk.p' , 'rb') as f:
        comein=pickle.load(f)
    tableP = comein[0]

    with open('./tableS_hk.p' , 'rb') as f:
        comein=pickle.load(f)
    tableS = comein[0]

    stations=[]
    with open('station_master.csv' ,'r') as f:
        comein = csv.reader(f, delimiter='|')
        line_count=0
        for row in comein:
            if line_count==0:
                title=row[0].split(',')
                line_count=line_count+1
            else:
                stations.append(row[0].split(','))
                line_count=line_count+1

    area_index=title.index('area')
    net_index=title.index('net')
    name_index=title.index('sta')
    z_index=title.index('z')
    e_index=title.index('e')
    n_index=title.index('n')
    lat_index=title.index('lat')
    lon_index=title.index('lon')
    elev_index=title.index('elev')

    with open('station_correction', 'r') as f:
        comein = f.readlines()

    station_correction = {'channel_name': 'value'}
    for i in comein:
        j = i.split(',')
        station_correction[j[0]] = float(j[1])

    station_use=[]
    for row in stations:
        if row[area_index]==area:
            station_use.append(row)

    lats=np.arange(studyarea[0],studyarea[0]+studyarea[2],latgrid)
    lons=np.arange(studyarea[1],studyarea[1]+studyarea[3],longrid)
    deps=np.arange(studydepth[0],studydepth[1]+depgrid,depgrid)
    studygrids=[]
    for i in lats:
        for j in lons:
            for k in deps:
                studygrids.append([i,j,k])

    print(len(studygrids))
    pickle.dump([studygrids],open('studygrids.p','wb'))

    deps_for_locate=np.arange(studydepth_for_locate[0],studydepth_for_locate[1]+depgrid,depgrid)
    studygrids_for_locate=[]
    for i in lats:
        for j in lons:
            for k in deps_for_locate:
                studygrids_for_locate.append([i,j,k])

    totalday = int(float(time_end['day']) - float(time_begin['day']) + 1)
    for date in range(0, totalday):

        day = int(float(time_begin['day']) + date)
        if day < 10:
            day='0' + str(day)
        else:
            day=str(day)

        stlas=[]
        stlos=[]
        stz=[]

        v=[]
        h1=[]
        h2=[]

        xmlfiles_v=[]
        xmlfiles_h1=[]
        xmlfiles_h2=[]

        st_correct_e = []
        st_correct_n = []

        for i in station_use:
            name1 = datadir + time_begin['month'] + '/' + day + '/' + time_begin[
                'year'] + time_begin['month'] + day + '.' + i[net_index] + '.' + i[name_index] + '..' + i[
                        z_index] + '.mseed'

            e1 = datadir + time_begin['month'] + '/' + day + '/' + time_begin[
                'year'] + time_begin['month'] + day + '.' + i[net_index] + '.' + i[name_index] + '..' + i[
                        e_index] + '.mseed'

            n1 = datadir + time_begin['month'] + '/' + day + '/' + time_begin[
                'year'] + time_begin['month'] + day + '.' + i[net_index] + '.' + i[name_index] + '..' + i[
                     n_index] + '.mseed'


            if Path(name1).is_file() == True and Path(e1).is_file() == True and Path(n1).is_file() == True:
                channelz=read(name1)
                channele=read(e1)
                channeln=read(n1)
                v.append(channelz)
                h1.append(channele)
                h2.append(channeln)
                stlas.append(float(i[lat_index]))
                stlos.append(float(i[lon_index]))
                stz.append(float(i[elev_index]))
                inv = obspy.read_inventory(xmldir +i[net_index] + '.' + i[name_index] + '.' + i[z_index] + '.xml', format='STATIONXML')
                xmlfiles_v.append(inv)
                inv = obspy.read_inventory(xmldir +i[net_index] + '.' + i[name_index] + '.' + i[e_index] + '.xml', format='STATIONXML')
                xmlfiles_h1.append(inv)
                inv = obspy.read_inventory(xmldir +i[net_index] + '.' + i[name_index] + '.' + i[n_index] + '.xml', format='STATIONXML')
                xmlfiles_h2.append(inv)
                try:
                    st_correct_e.append(station_correction[i[name_index] + '.' + i[net_index] + '.E'])
                except KeyError:
                    st_correct_e.append(0)

                try:
                    st_correct_n.append(station_correction[i[name_index] + '.' + i[net_index] + '.N'])
                except KeyError:
                    st_correct_n.append(0)

        '''
        epidis = []
        for j,k in zip(stlas, stlos):
            epidis.append(distance.dis(j,k,35.7695, -117.599333))
        '''

        traveldis=[]
        for i in studygrids:
            a=[]
            for j,k in zip(stlas,stlos):
                a.append(distance.dis(j,k,i[0],i[1]))

            traveldis.append(a)

        traveldis_for_locate=[]
        for i in studygrids_for_locate:
            a=[]
            for j,k in zip(stlas,stlos):
                a.append(distance.dis(j,k,i[0],i[1]))

            traveldis_for_locate.append(a)

        ptraveltimes=[]
        straveltimes=[]
        for i in range(0,len(traveldis)):
            if i%100==0:
                print(i)
            a=[]
            b=[]
            for j in range(0, len(traveldis[i])):
                timeP = taupz.taupz(tableP, tableS, studygrids[i][2], traveldis[i][j],'P', stz[j])
                a.append(timeP)
                timeS = taupz.taupz(tableP, tableS, studygrids[i][2], traveldis[i][j],'S', stz[j])
                b.append(timeS)

            ptraveltimes.append(a)
            straveltimes.append(b)

        ptraveltimes = np.array(ptraveltimes)
        straveltimes = np.array(straveltimes)

        ptraveltimes_for_locate=[]
        straveltimes_for_locate=[]
        for i in range(0,len(traveldis_for_locate)):
            if i%100==0:
                print(i)
            a=[]
            b=[]
            for j in range(0, len(traveldis_for_locate[i])):
                timeP = taupz.taupz(tableP, tableS, studygrids_for_locate[i][2], traveldis_for_locate[i][j],'P', stz[j])
                a.append(timeP)
                timeS = taupz.taupz(tableP, tableS, studygrids_for_locate[i][2], traveldis_for_locate[i][j],'S', stz[j])
                b.append(timeS)

            ptraveltimes_for_locate.append(a)
            straveltimes_for_locate.append(b)

        ptraveltimes_for_locate = np.array(ptraveltimes_for_locate)
        straveltimes_for_locate = np.array(straveltimes_for_locate)

        stations_plot.p(lat0=min(stlas)-0.5,lat=max(stlas)+0.5,lon0=min(stlos)-1,lon=max(stlos)+1,stlas=stlas,stlos=stlos,name= outputdir +'stations.pdf',marker='^',color='g')

        lat0=min(stlas)-0.5
        lat=max(stlas)+0.5
        lon0=min(stlos)-1
        lon=max(stlos)+1
        '''
        m = Basemap(llcrnrlon=lon0,llcrnrlat=lat0,urcrnrlon=lon,urcrnrlat=lat,projection='cass',lat_0=(lat0+lat)/2,lon_0=(lon0+lon)/2)
    
        m.drawcoastlines()
        m.drawparallels(np.arange(lat0, lat, (lat-lat0)/4), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(lon0, lon, (lon-lon0)/4), labels=[0, 0, 0, 1])
        m.drawmapboundary()
    
        x, y = m(stlos, stlas)
        m.scatter(x, y, 40, color='g', marker='^')
        #x, y = m(-129.48, 49.25)
        #m.scatter(x, y, 100, color='r', marker='*')
    
        plt.savefig(outputdir +'stations.pdf')
        '''
        #######################################################################

        nsta=len(v)

        for i in range(0,nsta):
            '''
            if len(v[i])>1:
                for j in range(0,len(v[i])):
                    v[i][j].stats.sampling_rate=round(v[i][j].stats.sampling_rate)
                v[i].merge(method=1, fill_value='interpolate')
            v[i]=v[i][0]
            '''
            v[i]=v[i][0]
            if v[i].stats.sampling_rate != sr:
                v[i].resample(sr)

            '''
            if len(h1[i])>1:
                for j in range(0,len(h1[i])):
                    h1[i][j].stats.sampling_rate=round(h1[i][j].stats.sampling_rate)
                h1[i].merge(method=1, fill_value='interpolate')
            h1[i]=h1[i][0]
            '''
            h1[i]=h1[i][0]
            if h1[i].stats.sampling_rate != sr:
                h1[i].resample(sr)

            '''
            if len(h2[i])>1:
                for j in range(0,len(h2[i])):
                    h2[i][j].stats.sampling_rate=round(h2[i][j].stats.sampling_rate)
                h2[i].merge(method=1, fill_value='interpolate')
            h2[i]=h2[i][0]
            '''
            h2[i]=h2[i][0]
            if h2[i].stats.sampling_rate != sr:
                h2[i].resample(sr)

        #duration=int((float(time_end['day'])-float(time_begin['day']) - 1)*24 + 24-float(time_begin['hour']) + float(time_end['hour']))
        if day == time_end['day']:
            duration = int(float(time_end['hour']))
        elif day == time_begin['day']:
            duration = 24 - int(float(time_begin['hour']))
        else:
            duration = 24

        if day == time_end['day'] and day == time_begin['day']:
            duration = int(float(time_end['hour'])) - int(float(time_begin['hour']))

        for runhour in range(0,duration):

            if day == time_begin['day']:
                runh = runhour
            else:
                runh = 24 - int(float(time_begin['hour'])) + (date-1)*24 + runhour

            if int(float(time_begin['hour'])+runh)%24 == 23:
                vborrow = []
                h1borrow = []
                h2borrow = []
                if int(float(day)+1) < 10:
                    dayp = '0' + str(int(float(day)+1))
                else:
                    dayp = str(int(float(day)+1))
                for i in station_use:
                    name1 = datadir + time_begin['month'] + '/' + dayp + '/' + time_begin[
                        'year'] + time_begin['month'] + dayp + '.' + i[net_index] + '.' + i[name_index] + '..' + i[
                                z_index] + '.mseed'

                    e1 = datadir + time_begin['month'] + '/' + dayp + '/' + time_begin[
                        'year'] + time_begin['month'] + dayp + '.' + i[net_index] + '.' + i[name_index] + '..' + i[
                             e_index] + '.mseed'

                    n1 = datadir + time_begin['month'] + '/' + dayp + '/' + time_begin[
                        'year'] + time_begin['month'] + dayp + '.' + i[net_index] + '.' + i[name_index] + '..' + i[
                             n_index] + '.mseed'

                    if Path(name1).is_file() == True and Path(e1).is_file() == True and Path(n1).is_file() == True:
                        channelz = read(name1)
                        channele = read(e1)
                        channeln = read(n1)
                        vborrow.append(channelz[0])
                        h1borrow.append(channele[0])
                        h2borrow.append(channeln[0])

                for i in range(len(vborrow)):
                    if vborrow[i].stats.sampling_rate != sr:
                        vborrow[i].resample(sr)
                    if h1borrow[i].stats.sampling_rate != sr:
                        h1borrow[i].resample(sr)
                    if h2borrow[i].stats.sampling_rate != sr:
                        h2borrow[i].resample(sr)

            t1 = UTCDateTime(time_begin['year']+'-'+time_begin['month']+'-'+str(int(float(time_begin['day'])+np.floor(float(time_begin['hour'])+runh)/24))+'T'+str(int(float(time_begin['hour'])+runh)%24)+':00:00')

            vfilter = []
            h1filter = []
            h2filter = []

            for i in v:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in vborrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.detrend()
                    ii.filter(type='bandpass', freqmin=scanlowf, freqmax=scanhighf, corners=3, zerophase=True)
                    ii.trim(t1, t1 + (process_unit + overlap) * 60)
                    vfilter.append(ii.data)
                except NotImplementedError:
                    vfilter.append(np.ones(points))

            for i in h1:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour']) + runh) % 24 == 23:
                    for look in h1borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.detrend()
                    ii.filter(type='bandpass', freqmin=scanlowf, freqmax=scanhighf, corners=3, zerophase=True)
                    ii.trim(t1, t1 + (process_unit + overlap) * 60)
                    h1filter.append(ii.data)
                except NotImplementedError:
                    h1filter.append(np.ones(points))

            for i in h2:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour']) + runh) % 24 == 23:
                    for look in h2borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.detrend()
                    ii.filter(type='bandpass', freqmin=scanlowf, freqmax=scanhighf, corners=3, zerophase=True)
                    ii.trim(t1, t1 + (process_unit + overlap) * 60)
                    h2filter.append(ii.data)
                except NotImplementedError:
                    h2filter.append(np.ones(points))

            '''
            epi=[-129.48, 49.25]
            epidis=[]
            for i,j in zip(stlas, stlos):
                epidis.append(geodis.geodis([[epi[1],epi[0]],[i,j]]))
        
            for i in range(0,len(vfilter)):
                plt.plot(vfilter[i]/max(vfilter[i])*5+epidis[i])
            plt.show()
            '''

            vn = []
            hn = []
            vkur = []
            hkur = []

            if psseparate == 'on':
                onlyP=[]
                onlyS=[]
                number=0
                for i,j,k in zip(h1filter, h2filter, vfilter):
                    number = number + 1
                    print(number)
                    try:
                        [P , S, LIN , COS] = psseparation(np.array([i,j,k]), 80)
                        onlyP.append(np.array(P))
                        onlyS.append(np.array(S))
                    except UnboundLocalError:
                        onlyP.append(np.ones(points))
                        onlyS.append(np.ones(points))

                for i in onlyP:
                    i=i-np.mean(i)
                    a=abs(i)
                    if math.isnan(np.median(a)) == True or np.median(a) == 0:
                        a = np.ones(points)
                    if len(a) < points:
                        a = np.ones(points)
                    if len(a) > points:
                        a=a[0: points]

                    for j in range(0,process_unit+overlap):
                        a[j *60* sr:(j + 1)*60*sr]=a[j*60*sr:(j+1)*60*sr]/np.median(a[j*60*sr:(j+1)*60*sr])
                    a=a**(1/root)
                    vn.append(a)

                for i in onlyS:
                    i=i-np.mean(i)
                    a=abs(i)
                    if math.isnan(np.median(a)) == True or np.median(a) == 0:
                        a = np.ones(points)
                    if len(a) < points:
                        a = np.ones(points)
                    if len(a) > points:
                        a=a[0: points]

                    for j in range(0,process_unit+overlap):
                        a[j *60* sr:(j + 1)*60*sr]=a[j*60*sr:(j+1)*60*sr]/np.median(a[j*60*sr:(j+1)*60*sr])
                    a=a**(1/root)
                    hn.append(a)
            else:
                count=0
                for i in vfilter:
                    a=abs(i)
                    if math.isnan(np.median(a)) == True or np.median(a) == 0:
                        a = np.ones(points)
                    if len(a) < points:
                        a = np.ones(points)
                    if len(a) > points:
                        a=a[0: points]

                    for j in range(0,process_unit+overlap):
                        a[j *60* sr:(j + 1)*60*sr]=a[j*60*sr:(j+1)*60*sr]/np.median(a[j*60*sr:(j+1)*60*sr])
                    a=a**(1/root)
                    vn.append(a)

                for i,j in zip(h1filter, h2filter):

                    if math.isnan(np.median(i)) == True or np.median(i) == 0:
                        i = np.ones(points)
                    if len(i) < points:
                        i = np.ones(points)
                    if len(i) > points:
                        i=i[0: points]

                    if math.isnan(np.median(j)) == True or np.median(j) == 0:
                        j = np.ones(points)
                    if len(j) < points:
                        j = np.ones(points)
                    if len(j) > points:
                        j=j[0: points]

                    a=(i**2+j**2)**0.5

                    for j in range(0,process_unit+overlap):
                        a[j *60* sr:(j + 1)*60*sr]=a[j*60*sr:(j+1)*60*sr]/np.median(a[j*60*sr:(j+1)*60*sr])
                    a=a**(1/root)
                    hn.append(a)

            if kurscan == 'on':
                for i in range(0, nsta):
                    print(i)
                    data = vfilter[i]
                    if len(data) < points:
                        data = np.ones(points)
                    if len(data) > points:
                        data = data[0: points]
                    vv = np.zeros(len(data))
                    for j in range(0, len(data)):
                        if j > sr * 5:
                            kur = mykurtosis(data[j - sr*4:j])
                            vv[j] = kur

                    cf = np.zeros(points)
                    for j in range(1, points):
                        if vv[j] - vv[j - 1] > 0:
                            cf[j] = vv[j] - vv[j - 1]
                    vkur.append(cf)

                for i in range(0, nsta):
                    print(i)
                    vv = []
                    data1 = h1filter[i]
                    data2 = h2filter[i]
                    if len(data1) < points or len(data2) < points:
                        data1 = np.ones(points)
                        data2 = np.ones(points)
                    if len(data1) > points:
                        data1 = data1[0: points]
                    if len(data2) > points:
                        data2 = data2[0: points]
                    vv = np.zeros(len(data1))
                    for j in range(0, len(data1)):
                        if j > sr * 5:
                            kur1 = mykurtosis(data1[j - sr*4:j])
                            kur2 = mykurtosis(data2[j - sr*4:j])
                            kur = (kur1 + kur2) / 2
                            vv[j] = kur

                    cf = np.zeros(points)
                    for j in range(1, points):
                        if vv[j] - vv[j - 1] > 0:
                            cf[j] = vv[j] - vv[j - 1]
                    hkur.append(cf)

            if brscan == 'on':
                brp = ssa.med_scan_mul(ptraveltimes, np.array(vn), sr, win, step, processes)
                brmapp = ((brp / (win * sr)) ** 0.5 / nsta) ** root
                pickle.dump([brmapp], open(outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'ssabrmap_P.p', 'wb'))

                brs = ssa.med_scan_mul(straveltimes, np.array(hn), sr, win, step, processes)
                brmaps = ((brs / (win * sr)) ** 0.5 / nsta) ** root
                pickle.dump([brmaps], open(outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'ssabrmap_S.p', 'wb'))

                '''
                shape_s = np.shape(brmaps)
                brmapp = brmapp[:, 0:shape_s[1]]
                brmap = np.multiply(brmaps, brmapp)
        
                pickle.dump([brmap], open(outputdir+str(int(float(time_begin['hour'])+runh))+'ssabrmap.p', 'wb'))
                '''
            else:
                brp = ssa.med_scan(ptraveltimes, np.array(vn), sr, win, step)
                brmapp = ((brp / (win * sr)) ** 0.5 / nsta) ** 0
                pickle.dump([brmapp],
                            open(outputdir + day + str(int(float(time_begin['hour']) + runh) % 24) + 'ssabrmap_P.p', 'wb'))

                brs = ssa.med_scan(straveltimes, np.array(hn), sr, win, step)
                brmaps = ((brs / (win * sr)) ** 0.5 / nsta) ** 0
                pickle.dump([brmaps],
                            open(outputdir + day + str(int(float(time_begin['hour']) + runh) % 24) + 'ssabrmap_S.p', 'wb'))

            if kurscan == 'on':
                brp = ssa.kur_scan(ptraveltimes, np.array(vkur), sr, win, step)
                kurmapp = brp / nsta
                pickle.dump([kurmapp], open(outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'ssabrmap_cfmax_P.p', 'wb'))

                brs = ssa.kur_scan(straveltimes, np.array(hkur), sr, win, step)
                kurmaps = brs / nsta
                pickle.dump([kurmaps], open(outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'ssabrmap_cfmax_S.p', 'wb'))

                '''
                shape_s = np.shape(kurmaps)
                kurmapp = kurmapp[:, 0:shape_s[1]]
                kurmap = np.multiply(kurmaps, kurmapp)
        
                pickle.dump([kurmap], open(outputdir+str(int(float(time_begin['hour'])+runh))+'ssabrmap_cfmax.p', 'wb'))
                '''

            shape_s = np.shape(brmaps)
            brmapp = brmapp[:, 0:shape_s[1]]
            brmap = np.multiply(brmaps, brmapp)

            if kurscan == 'on':
                shape_s = np.shape(brmaps)
                kurmapp = kurmapp[:, 0:shape_s[1]]
                kurmaps = kurmaps[:, 0:shape_s[1]]
                kurmap = np.multiply(kurmaps, kurmapp)

                brmap = np.multiply(brmap, kurmap)

            brmax = []
            for i in range(0, len(brmap[0])):
                brmax.append(max(brmap[:, i]))

            plt.clf()
            plt.semilogy(brmax)
            # plt.show()
            plt.savefig(outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'brmax.pdf')

            peaktimes = detect_peaks(x=brmax, mph=1, mpd=0)
            print(peaktimes)

            peakheight = []
            for i in peaktimes:
                peakheight.append((brmax[i], i))

            order = sorted(peakheight, key=itemgetter(0), reverse=True)

            peaks = []
            for i in order:
                peaks.append(brmap[:, i[1]])

            pp = []
            for i in peaks:
                m = max(i)
                p = [j for j, k in enumerate(i) if k == m]
                pp.append(p)
            print(pp)

            evlas = []
            evlos = []
            evdeps = []
            for i in pp:
                evlas.append(studygrids[i[0]][0])
                evlos.append(studygrids[i[0]][1])
                evdeps.append(studygrids[i[0]][2])
            '''
            print(evlas)
            print(evlos)
            print(evdeps)
            pickle.dump([pp, evlas, evlos, evdeps], open(outputdir+str(int(float(time_begin['hour'])+runh))+'SSAlocation.p', 'wb'))
            '''

            vraw=[]
            h1raw=[]
            h2raw=[]
            for i in v:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in vborrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.detrend()
                    ii.trim(t1, t1 + (process_unit + overlap) * 60)
                except NotImplementedError:
                    ii.data=np.zeros(points)

                vraw.append(ii)


            for i in h1:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in h1borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.detrend()
                    ii.trim(t1, t1 + (process_unit + overlap) * 60)
                except NotImplementedError:
                    ii.data = np.zeros(points)

                h1raw.append(ii)

            for i in h2:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in h2borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.detrend()
                    ii.trim(t1, t1 + (process_unit + overlap) * 60)
                except NotImplementedError:
                    ii.data = np.zeros(points)

                h2raw.append(ii)


            [ponsets_final, sonsets_final] = picking_seisbench.pick(points, outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'_', nametag, nsta,order, pp, ptraveltimes, straveltimes, step, win, sr, dstop, vraw, h1raw, h2raw, savefigure, overlap, trained_model, aipick_threshold)

            pickle.dump([ponsets_final], open(outputdir +day+str(int(float(time_begin['hour'])+runh)%24)+'ponsets_final_' + nametag + '.p', 'wb'))
            pickle.dump([sonsets_final],open(outputdir +day+str(int(float(time_begin['hour'])+runh)%24)+ 'sonsets_final_' + nametag + '.p','wb'))

            [events, MAXI, catalog] = locatePS.MAXI_locate(ponsets_final, sonsets_final, ptraveltimes_for_locate, straveltimes_for_locate, stlas, stlos, stz, lowq, highq, studygrids_for_locate, Q_threshold, terr, outlier, tableP, tableS, latgrid, longrid, depgrid, mindepgrid, minimprove)

            pickle.dump([events], open(outputdir +day+str(int(float(time_begin['hour'])+runh)%24)+ 'MAXI_PRED' + nametag + '.p', 'wb'))
            pickle.dump([MAXI],open(outputdir +day+str(int(float(time_begin['hour'])+runh)%24)+ 'MAXIvalue' + nametag +'.p','wb'))
            pickle.dump([catalog],open(outputdir +day+str(int(float(time_begin['hour'])+runh)%24)+ 'cataloghigh' + nametag +'.p','wb'))

            vraw = []
            h1raw = []
            h2raw = []
            for i in v:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour']) + runh) % 24 == 23:
                    for look in vborrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.detrend()
                    ii.trim(t1, t1 + (process_unit + overlap) * 60)
                except NotImplementedError:
                    ii.data = np.zeros(points)

                vraw.append(ii)

            for i in h1:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour']) + runh) % 24 == 23:
                    for look in h1borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.detrend()
                    ii.trim(t1, t1 + (process_unit + overlap) * 60)
                except NotImplementedError:
                    ii.data = np.zeros(points)

                h1raw.append(ii)

            for i in h2:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour']) + runh) % 24 == 23:
                    for look in h2borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.detrend()
                    ii.trim(t1, t1 + (process_unit + overlap) * 60)
                except NotImplementedError:
                    ii.data = np.zeros(points)

                h2raw.append(ii)

            magnitude = calc_mag.CISN_local_magnitude(sr, stz, stlas, stlos, xmlfiles_h1, xmlfiles_h2, st_correct_e, st_correct_n, catalog, events, sonsets_final, h1raw, h2raw, points)
            pickle.dump([magnitude],open(outputdir +day+str(int(float(time_begin['hour'])+runh)%24)+ 'magnitude' + nametag +'.p','wb'))

            high = []
            for i in magnitude:
                if len(i) == 8 and i[6] >= highq * 2:
                    high.append(i)

            cataloghigh = sorted(high, key=itemgetter(0), reverse=False)
            for i in range(0, len(cataloghigh)):
                cataloghigh[i][0] = str(int(np.floor(cataloghigh[i][0] / 60))) + ':' + str(cataloghigh[i][0] % 60)

            f = open(outputdir + day + str(int(float(time_begin['hour'])+runh)%24) + 'cataloghigh_' + nametag + '.txt', 'w')
            for i in cataloghigh:
                for j in i:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
            f.close()

            evlashigh = []
            evloshigh = []
            evdepshigh = []

            evlaslow = []
            evloslow = []
            evdepslow = []

            for i in magnitude:
                if i[5] > 0:
                    evlashigh.append(i[1])
                    evloshigh.append(i[2])
                    evdepshigh.append(i[3])
                else:
                    evlaslow.append(i[1])
                    evloslow.append(i[2])
                    evdepslow.append(i[3])

            stations_plot.p(lat0=studyarea[0], lat=studyarea[0] + studyarea[2], lon0=studyarea[1], lon=studyarea[1] + studyarea[3], stlas=evlashigh,
                            stlos=evloshigh, name=outputdir + day + str(int(float(time_begin['hour'])+runh)%24) + 'cataloghigh_' + nametag + '.pdf', marker='.',
                            color='r')
            stations_plot.p(lat0=studyarea[0], lat=studyarea[0] + studyarea[2], lon0=studyarea[1], lon=studyarea[1] + studyarea[3], stlas=evlaslow,
                            stlos=evloslow, name=outputdir + day + str(int(float(time_begin['hour'])+runh)%24) + 'cataloglow_' + nametag + '.pdf', marker='.',
                            color='b')


if __name__ == "__main__":
    main()
