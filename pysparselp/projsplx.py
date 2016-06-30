import numpy as np

def projsplx(y):
    """ project a set of  n-dim vectors y to the simplex Dn
     Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}
     
     adapted from code by Xiaojing Ye to to make it vectorialized 
     which help to get a faster code in interpreted languages
    
     (c) Xiaojing Ye
     xyex19@gmail.com
    
     Algorithm is explained as in the linked document
     http://arxiv.org/abs/1101.6081
     or
     http://ufdc.    s = np.sort(y,axis=1)[:,::-1] 
ufl.edu/IR00000353/
     Jan. 14, 2011.
     
     """
    assert(y.ndim==2)
    m = y.shape[1]
    s = np.sort(y,axis=1)[:,::-1] 
    x=np.zeros(y.shape)

    for i in range(y.shape[0]):
        tmpsum = 0
        bget = False
        for j in range(m-1):
            tmpsum = tmpsum + s[i,j];
            tmax = (tmpsum - 1)/(j+1);
            if tmax >= s[i,j+1]:
                bget = True
                break
        if not bget:
            tmax = (tmpsum + s[i,m-1] -1)/m
        for j in range(m):   
            x[i,j] = max(y[i,j]-tmax,0)
    return x
