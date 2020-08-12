# This module is to plot the basic map and stations and epicenter

def p(lat0=0,lon0=0,lat=1,lon=1,stlas= [47.761659, 48.7405, 49.755100, 45.737167],stlos= [12.864466, 11.8671, 10.849660, 14.795714],name='stations.pdf',marker='^',color='g'):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    
    plt.clf()
    #plt.ion()

    m = Basemap(llcrnrlon=lon0,llcrnrlat=lat0,urcrnrlon=lon,urcrnrlat=lat,projection='cass',lat_0=(lat0+lat)/2,lon_0=(lon0+lon)/2)

    m.drawcoastlines()
    m.drawparallels(np.arange(lat0, lat, (lat-lat0)/3), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(lon0, lon, 1), labels=[0, 0, 0, 1])
    m.drawmapboundary()

    x, y = m(stlos, stlas)
    m.scatter(x, y, 40, color=color, marker=marker)

    plt.savefig(name)
