from math import radians, cos, sin, asin,acos, sqrt, pi

def dis(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    #a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    #c = 2 * asin(sqrt(a))/pi*180
    a=sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(dlon)
    if a>1:
        a=1
    if a<-1:
        a=-1
    c=acos(a)/pi*180
    return c



