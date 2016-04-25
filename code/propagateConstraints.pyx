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
@cython.cdivision(True)
def propagateConstraints(list_changed_var,\
			np.ndarray[cDOUBLE, ndim=1] x_l,\
			np.ndarray[cDOUBLE, ndim=1] x_u,\
			np.ndarray[cINT32, ndim=1] Acsc_indices,\
			np.ndarray[cINT32, ndim=1] Acsr_indices,\
			np.ndarray[cINT32, ndim=1] Acsr_indptr,\
			np.ndarray[cINT32, ndim=1] Acsc_indptr,\
			np.ndarray[cDOUBLE, ndim=1] Acsr_data,
			np.ndarray[cDOUBLE, ndim=1] b_lower,
			np.ndarray[cDOUBLE, ndim=1] b_upper,
			list back_ops,\
			int nb_iter=1000):
	
	
	cdef int itere ,i,j,k,indptr_j,nb_var
	cdef double interval_l,interval_u,v,tol,n_u,n_l
	cdef vector[int] list_changed_var_cpp= list_changed_var
	#cdef set[int] list_changed_var_cpp= list_changed_var
	cdef set[int]  list_constraints_to_check_cpp
	#cdef vector[int]  list_constraints_to_check_cpp
	
	#cdef np.ndarray[cINT32, ndim=1] Acsc_indices	
	#cdef np.ndarray[cINT32, ndim=1] Acsr_indices
	#cdef np.ndarray[cINT32, ndim=1] Acsr_indptr
	#cdef np.ndarray[cINT32, ndim=1] Acsc_indptr
	#cdef np.ndarray[cDOUBLE, ndim=1] Acsr_data
	
	#Acsc_indices=Acsc.indices.astype(np.int32)
	#Acsr_indices=Acsr.indices.astype(np.int32)
	#Acsr_indices=Acsr.indices.astype(np.int32)
	#Acsr_indptr =Acsr.indptr
	#Acsc_indptr =Acsc.indptr
	#Acsr_data=Acsr.data.astype(np.float)
	


	tol=1e-5 # to cope with small errors
	for iter in range(nb_iter):
		#print '%d variable fixed '% np.sum(x_l==x_u)
		#list_changed_var=np.unique(list_changed_var)
		#if len(list_changed_var_cpp)==0:
		if list_changed_var_cpp.size()==0:
			break
		
		#list_constraints_to_check=[]
		#for i in list_changed_var_cpp:
			##to_add=np.nonzero(Acsc[:,i])[0]
			#to_add2=Acsc.indices[Acsc.indptr[i]:Acsc.indptr[i+1]]
			##assert(np.all(to_add==to_add2))
			#list_constraints_to_check.append(to_add2)
		#list_constraints_to_check_cpp=np.unique(np.hstack(list_constraints_to_check))
		list_constraints_to_check_cpp.clear()
		for i in list_changed_var_cpp:
			for j in range(Acsc_indptr[i],Acsc_indptr[i+1]):
				list_constraints_to_check_cpp.insert(Acsc_indices[j])
				#list_constraints_to_check_cpp.push_back(Acsc_indices[j])
			
		
		
		#list_changed_var=[]
		#list_changed_var_cpp=[]
		list_changed_var_cpp.clear()
		#list_constraints_to_check=np.arange(Acsr.shape[0])
		for j in list_constraints_to_check_cpp:
			#line=Acsr[j,:]# very slow...
			#indices=line.indices
			#data=line.data
			indptr_j=Acsr_indptr[j]
			nb_var=Acsr_indptr[j+1]-Acsr_indptr[j]
			#indices=Acsr.indices[Acsr.indptr[j]:Acsr.indptr[j+1]]
			#data=Acsr.data[Acsr.indptr[j]:Acsr.indptr[j+1]]

			
			interval_l=0
			interval_u=0
			for k in range(nb_var):
				#i=indices[k]
				#v=data[k]
				i=Acsr_indices[k+indptr_j]
				v=Acsr_data[k+indptr_j]
				if v>0:
					interval_u+=v*x_u[i]
					interval_l+=v*x_l[i]
				else: 
					interval_l+=v*x_u[i]
					interval_u+=v*x_l[i]				
					
			if interval_u<b_lower[j] or interval_l>b_upper[j]:
				return 0,j
					
			for k in range(nb_var):
				#i=indices[k]
				#v=data[k]
				i=Acsr_indices[k+indptr_j]
				v=Acsr_data[k+indptr_j]
				if v>0:
					
					n_u=floor(tol+(b_upper[j]-interval_l+v*x_l[i])/v)					
					n_l=ceil(-tol+(b_lower[j]-interval_u+v*x_u[i])/v)
				else: 
					n_u=floor(tol+(b_lower[j]-interval_u+v*x_l[i])/v)
					n_l=ceil(-tol+(b_upper[j]-interval_l+v*x_u[i])/v)
									
				changed=False	
				if n_u<x_u[i]:
					back_ops.append((1,i,x_u[i]))# save previous information for future backtracking
					x_u[i]=n_u
					changed=True
				if n_l>x_l[i]:	
					back_ops.append((0,i,x_l[i]))
					x_l[i]=n_l
					changed=True
				if changed:
					#list_changed_var_cpp.append(i)
					list_changed_var_cpp.push_back(i)
					#list_changed_var_cpp.insert(i)
					#assert(j in list_constraints_to_check2)
					
					
					
	#print '%d variable fixed '% np.sum(x_l==x_u)
	return 1,None
