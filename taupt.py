def t(model, dep, dis, phase):
    a=model.get_travel_times(dep,dis,[phase])
    return(a[0].time)