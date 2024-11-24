def geodis(a):
    b=a[0]
    c=a[1]
    from geopy.distance import distance
    dis=distance(b,c).km
    return(dis)
