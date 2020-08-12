def geodis(a):
    b=a[0]
    c=a[1]
    from geopy.distance import vincenty
    dis=vincenty(b,c).km
    return(dis)
