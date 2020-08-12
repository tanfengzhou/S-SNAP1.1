
from obspy.taup import taup_create
#taup_create.build_taup_model('/Users/tanfengzhou/Documents/Research/2019california/aftershock/scec1d_new.tvel')
from obspy.taup import TauPyModel
model = TauPyModel(model='scec1d')
import numpy as np
import pickle

dis = np.arange(0,2.4,0.003)      # degree
dep = np.arange(0,40,0.3)       # km

tableP=np.zeros([len(dis),len(dep)])
tableS=np.zeros([len(dis),len(dep)])

for i in range(0, len(dis)):
    print(dis[i])
    for j in range(0, len(dep)):
        times = model.get_travel_times(dep[j], dis[i], ['p'])
        if len(times) > 0:
            t = times[0].time
        else:
            times = model.get_travel_times(dep[j], dis[i], ['P'])
            if len(times) == 1:
                t = times[0].time
            else:
                takeoff = []
                for a in times:
                    takeoff.append(a.takeoff_angle)

                n = takeoff.index(max(takeoff))
                t = times[n].time

        tableP[i][j]=t
        tableS[i][j]=t*1.732

        
'''
for i in range(0, len(dis)):
    for j in range(0, len(dep)):
        tp = (dep[j]**2 + (dis[i]*111.20973427399475)**2) ** 0.5 / 5.5
        ts = tp * 1.73
        print(tp)
        tableP[i][j] = tp
        tableS[i][j] = ts
'''

pickle.dump([tableP],open('tableP_hk.p','wb'))
pickle.dump([tableS],open('tableS_hk.p','wb'))
