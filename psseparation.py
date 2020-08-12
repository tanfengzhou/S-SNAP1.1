def psseparation(data, win):
    # input three components, output P and S.
    # data should have the np.array structure. The first and second rows are h1 and h2 components, respectively. The third row is vertical.
    # win is sample amount, not time.
    import numpy as np
    from numba import jit

    @jit(nopython=True)
    def svdnumba(array):
        u, s, v = np.linalg.svd(array, full_matrices=False)
        return u,s,v

    P = []
    S = []
    LIN = []
    COS = []

    for i in range(0,len(data[0])-win+1):
        single = np.array([data[2,i:i+win],data[1,i:i+win],data[0,i:i+win]])
        #u, s, v = np.linalg.svd(single, 0, 1)
        u,s,v = svdnumba(single)
        #lin = s[0]/(s[0]+s[1]+s[2])
        lin = ((s[0]-s[1])**2 + (s[0]-s[2])**2 + (s[1]-s[2])**2)/(2*(s[0]+s[1]+s[2])**2)
        #lin = 1
        cos_th = abs(u[0,0]/(u[0,0]**2 + u[1,0]**2 + u[2,0]**2) ** 0.5)
        sin_th = (1 - cos_th**2) ** 0.5
        #print(lin, cos_th)
        LIN.append(lin)
        COS.append(cos_th)
        if i == 0:
            for j in range(0,round(win/2)+1):
                LIN.append(lin)
                COS.append(cos_th)
                P.append(abs(data[2, j]) * lin * cos_th)
                S.append((data[0, j] ** 2 + data[1, j] ** 2) ** 0.5 * lin * sin_th)
        else:
            P.append(abs(data[2, i + round(win / 2)]) * lin * cos_th)
            S.append((data[0, i] ** 2 + data[1, i + round(win / 2)] ** 2) ** 0.5 * lin * sin_th)


    for i in range(len(data[0])-win+1 + round(win/2), len(data[0])):
        #print(lin, cos_th)
        LIN.append(lin)
        COS.append(cos_th)
        P.append(abs(data[2,i]) * lin * cos_th)
        S.append((data[0, i] ** 2 + data[1, i] ** 2) ** 0.5 * lin * sin_th)

    return ([P, S, LIN, COS])
