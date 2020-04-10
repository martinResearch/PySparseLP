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

	
	

@cython.boundscheck(False) # turn of bounds-checking for entire function		
def GaussSeidel(A, np.ndarray[cDOUBLE, ndim=1] b,np.ndarray[cDOUBLE, ndim=1] x, int maxiter=3,double w=1,order=None,plotfunc=None,xopt=None):
# when w != 0 this implements a bounded version of http://en.wikipedia.org/wiki/Successive_over-relaxation
# can this algorithm be understood as a simple coordinate descent with the function |Ax-b|^2 under box constraints ? 
	
	assert(sparse.isspmatrix_csr(A))
	assert(A.dtype==np.float)
	cdef np.ndarray[cDOUBLE, ndim=1] D,invD
	D=A.diagonal()
	invD=1/D	
	
	#cdef np.ndarray[cDOUBLE, ndim=1] x
	#x=x0#.copy().astype(float)	
	#cdef np.ndarray[cDOUBLE, ndim=1] data=A.data	
	#cdef np.ndarray[cINT32, ndim=1] indptr=A.indptr
	#cdef np.ndarray[cINT32, ndim=1] indices=A.indices
	
	cdef cDOUBLE[:] data=A.data	
	cdef cINT32[:] indptr=A.indptr
	cdef cINT32[:] indices=A.indices
	

	
	cdef unsigned int k
	cdef double nv
	cdef double v
	cdef unsigned int i,j
	cdef unsigned int n=b.size
	if order==None:
		order=np.arange(n,dtype=np.uint32)
	else: 
		assert(order.dtype==np.uint32)
	
	cdef  np.ndarray[np.uint32_t, ndim=1] orderc=order
	
	if not xopt is None:
		curve=[]
	
	for _niter in range(maxiter):
		for j in range(n):
			i=orderc[j]
			
			#v=A[i,:].dot(x)
			#v=x[A.indices[A.indptr[i]:A.indptr[i+1]]].dot(A.data[A.indptr[i]:A.indptr[i+1]])
			v=0
			for k in range(<unsigned int> indptr[i],<unsigned int> indptr[i+1]):
				v+=x[<unsigned int> indices[k]]*data[k]
			nv=(b[i]-v+D[i]*x[i])*invD[i]			
			nv=w*nv+(1-w)*x[i]
			x[i]=nv

		if not plotfunc is None:
			plotfunc(x)
		if not xopt is None:
			curve.append(np.mean(np.abs(x-xopt)))
			
	if not xopt is None:
		return x,curve 
	else:
		return x



class boundedGaussSeidelClass:
	
	#cdef  np.ndarray invD
	
	def __init__(self,A):
		assert(sparse.isspmatrix_csr(A))
		assert(A.dtype==np.float)
		self.A=A
		D=A.diagonal()
		self.invD=1/D
		
	@cython.boundscheck(False)
	def solve(self,np.ndarray[cDOUBLE, ndim=1] b,np.ndarray[cDOUBLE, ndim=1] lower_bounds,np.ndarray[cDOUBLE, ndim=1] upper_bounds,cDOUBLE[:] x,int maxiter=3,double w=1,order=None):

	
		# turn of bounds-checking for entire function		
	       
		# when w != 0 this implements a bounded version of http://en.wikipedia.org/wiki/Successive_over-relaxation
		# can this algorithm be understood as a simple coordinate descent with the function |Ax-b|^2 under box constraints ? 
		       
		
		
		#cdef np.ndarray[cDOUBLE, ndim=1] x
		#x=x0#.copy().astype(float)	
		#cdef np.ndarray[cDOUBLE, ndim=1] data=A.data	
		#cdef np.ndarray[cINT32, ndim=1] indptr=A.indptr
		#cdef np.ndarray[cINT32, ndim=1] indices=A.indices
		
		cdef cDOUBLE[:] data=self.A.data	
		cdef cINT32[:] indptr=self.A.indptr
		cdef cINT32[:] indices=self.A.indices
		cdef np.ndarray[cDOUBLE, ndim=1] invD=self.invD
	
		
		cdef unsigned int k,l2
		cdef double nv
		cdef double v,l,u
		cdef unsigned int i,j
		cdef unsigned int n=b.size
		if order is None:
			order=np.arange(n,dtype=np.uint32)
		else: 
			assert(order.dtype==np.uint32)
		
		cdef  np.ndarray[np.uint32_t, ndim=1] orderc=order
		
		
		
		for _iter in range(maxiter):
			#for j in range(n):
				#i=orderc[j]
			for i in range(n):
				
				#v=A[i,:].dot(x)
				#v=x[A.indices[A.indptr[i]:A.indptr[i+1]]].dot(A.data[A.indptr[i]:A.indptr[i+1]])
				v=0
				for k in range(<unsigned int> indptr[i],<unsigned int> indptr[i+1]):
					l2=<unsigned int> indices[k]
					v+=x[l2]*data[k]
					#v+=x[i]*data[k]
				#nv=(b[i]-v)*invD[i]+x[i]		
				#v=w*nv+(1-w)*x[i]
				v=w*(b[i]-v)*invD[i]+x[i]
				l=lower_bounds[i]
				u=upper_bounds[i]
				if v<l:
					v=l
				elif v>u:
					v=u
				x[i]=v
		return x

if __name__ == "__main__":
	A=np.array([[16,3],[7,-11]])
	b=np.array([11,13])
	x0=np.array([1,1])
	bs=boundedGaussSeidelClass(A)
	bs.solve(b,-20,20,x0, maxiter=8)

