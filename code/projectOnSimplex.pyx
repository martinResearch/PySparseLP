# distutils: language = c++
import numpy as np
cimport numpy as np
cimport numpy as cnp
from libc.math cimport floor, ceil
from scipy import sparse
from scipy.sparse import diags
cimport cython
import scipy
from numpy.math cimport INFINITY
from libcpp.vector cimport vector
from libcpp.set cimport set

ctypedef np.int32_t cINT32
ctypedef np.double_t cDOUBLE

@cython.boundscheck(False)
@cython.wraparound(False) 
def projsplx(np.ndarray[cDOUBLE, ndim=2] y):
	""" project a set of  n-dim vectors y to the simplex Dn
	Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}
	
	adapted from code by to to make it vectorialized 
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
	
	s = np.sort(y,axis=1)[:,::-1] 
	cdef np.ndarray[cDOUBLE, ndim=2] x=np.zeros((y.shape[0],y.shape[1]),dtype=np.double)
	
	cdef int i, j,m
	cdef double tmpsum
	cdef int bget
	m = y.shape[1]
	
	for i in range(y.shape[0]):
		tmpsum = 0
		bget = 0
		for j in range(m-1):
			tmpsum = tmpsum + s[i,j];
			tmax = (tmpsum - 1)/(j+1);
			if tmax >= s[i,j+1]:
				bget = 1
				break
		if bget==0:
			tmax = (tmpsum + s[i,m-1] -1)/m
		for j in range(m):   
			x[i,j] = max(y[i,j]-tmax,0)
		
	assert(np.all(x>=0))
	assert(np.all(np.abs(np.sum(x,axis=1)-1)<1e-6))	
	return x
