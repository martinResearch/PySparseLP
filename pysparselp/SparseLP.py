# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------
#Copyright © 2016 Martin de la Gorce <martin[dot]delagorce[hat]gmail[dot]com>

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse
import scipy.ndimage
import sys
from pysparselp.ADMM import *
from pysparselp.ChambollePockPPD import *
from pysparselp.ChambollePockPPDAS import *
from pysparselp.ADMMBlocks import *
from pysparselp.DualGradientAscent import *
from pysparselp.DualCoordinateAscent import *
from pysparselp.MehrotraPDIP import mpcSol
#import cvxpy

solving_methods=["Mehrotra","ScipyLinProg","DualCoordinateAscent","DualGradientAscent","ChambollePockPPD",'ChambollePockPPDAS',"ADMM","ADMM2","ADMMBlocks"]
try:
	import cvxpy
	solving_methods.append("ECOS")
	solving_methods.append("SCS")	
except:
	print ('could not import cvxpy, the solvers ECOS and SCS will not be available')

	


def csr_matrix_append_row(A,n,cols,vals):
	A.blocks.append((A.shape[0],A.shape[0]))
	A._shape=(A.shape[0]+1,n) 
	A.indices=np.append(A.indices,cols.astype(A.indices.dtype))
	A.data=np.append(A.data,vals.astype(A.data.dtype))
	A.indptr=np.append(A.indptr,np.int32(A.indptr[-1]+cols.size))
	assert(A.data.size==A.indices.size)
	assert(A.indptr.size==A.shape[0]+1)
	assert(A.indptr[-1]==A.data.size)

def check_csr_matrix(A):
	assert(np.max(A.indices)<A.shape[1])
	assert(len(A.data)==len(A.indices))
	assert(len(A.indptr)==A.shape[0]+1)
	assert(np.all(np.diff(A.indptr)>=0))


def csr_matrix_append_rows(A,B):
	#A._shape=-A.shape[0],B._shape[1])	
	A.blocks.append((A.shape[0],A.shape[0]+B.shape[0]-1))
	A._shape=(A.shape[0]+B.shape[0],max(A.shape[1],B.shape[1])) 
	A.indices=np.append(A.indices,B.indices)
	A.data=np.append(A.data,B.data)
	A.indptr=np.append(A.indptr[:-1],A.indptr[-1]+B.indptr)
	
	assert(np.max(A.indices)<A.shape[1])
	assert(A.data.size==A.indices.size)
	assert(A.indptr.size==A.shape[0]+1)
	assert(A.indptr[-1]==A.data.size)

def empty_csr_matrix():
	A=scipy.sparse.csr_matrix((1,1),dtype=np.float)
	A._shape=(0*A._shape[0],0*A._shape[1]) #trick , because it would not let me create and empty matrix
	A.indptr=A.indptr[-1:]	
	return A

def unique_rows(data, prec=5):
	import numpy as np
	d_r = np.fix(data * 10 ** prec) / 10 ** prec + 0.0
	b = np.ascontiguousarray(d_r).view(np.dtype((np.void, d_r.dtype.itemsize * d_r.shape[1])))
	_, ia = np.unique(b, return_index=True)
	_, ic = np.unique(b, return_inverse=True)
	return np.unique(b).view(d_r.dtype).reshape(-1, d_r.shape[1]), ia, ic

class SparseLP():
	def __init__(self):
		# start writing the linear program 

		
		self.nb_variables=0
		self.variables_dict=dict()        
		self.upperbounds=np.empty((0),dtype=np.float)
		self.lowerbounds=np.empty((0),dtype=np.float)
		self.costsvector=np.empty((0),dtype=np.float)
		self.isinteger=np.empty((0),dtype=np.bool)
		self.Ainequalities=empty_csr_matrix()
		self.Ainequalities.__dict__['blocks']=[]
		self.B_lower=np.empty((0),dtype=np.float)
		self.B_upper=np.empty((0),dtype=np.float)
		self.Aequalities=empty_csr_matrix()
		self.Bequalities=np.empty((0),dtype=np.float)   
		self.Aequalities.__dict__['blocks']=[]
		self.solver='chambolle_pock'
		self.equalityConstraintNames=[]
		self.inequalityConstraintNames=[]
		self.solution=None
		
		
	def maxConstraintViolation(self,solution):
		types, lb, ub=self.getVariablesBounds()
		maxv=0
		maxv=max(maxv,np.max(lb-solution))
		maxv=max(maxv,np.max(solution-ub))
		if self.Aequalities.shape[0]>0:
			maxv=max(maxv,np.max(np.abs(self.Aequalities*solution-self.Bequalities)))
		if self.Ainequalities.shape[0]>0:
			if not self.B_upper is None:
				maxv=max(maxv,np.max(self.Ainequalities*solution-self.B_upper))
			if not self.B_lower is None:
				maxv=max(maxv,np.max(self.B_lower-self.Ainequalities*solution))
		return maxv		
		

	def checkSolution(self,solution,tol=1e-6):
		types, lb, ub=self.getVariablesBounds()
		valid=True
		if not lb is None:
			valid=valid&(np.max(lb-solution)<tol)
		if not ub is None:
			valid=valid&(np.max(solution-ub)<tol)
		if(not self.Aequalities is None)and self.Aequalities.shape[0]>0:
			valid=valid&(np.max(np.abs(self.Aequalities*solution-self.Bequalities))<tol)
		if (not self.Ainequalities is None)and self.Ainequalities.shape[0]>0:
			if not self.B_upper is None:
				valid=valid&(np.max(self.Ainequalities*solution-self.B_upper)<tol)
			if not self.B_lower is None:
				valid=valid&(np.max(self.B_lower-self.Ainequalities*solution)<tol)
		return valid
			
	def startConstraintName(self,name):
		if not (name is None or name==''):
			self.lastNameStart=name
			self.lastNameEqualityStart=self.nbEqualityConstraints()
			self.lastNameInequalityStart=self.nbInequalityConstraints()
		
	def nbEqualityConstraints(self):
		return self.Aequalities.shape[0]
		
	def nbInequalityConstraints(self):
		return self.Ainequalities.shape[0]
	def endConstraintName(self,name):
		
		if not (name is None or name==''):
			assert(self.lastNameStart==name)
			if self.nbEqualityConstraints()>self.lastNameEqualityStart:
				self.equalityConstraintNames.append({'name':name,'start':self.lastNameEqualityStart,'end':self.nbEqualityConstraints()-1})
			if self.nbInequalityConstraints()>self.lastNameInequalityStart:
				self.inequalityConstraintNames.append({'name':name,'start':self.lastNameInequalityStart,'end':self.nbInequalityConstraints()-1})	
	def getInequalityConstraintNameFromId(self,id):
		for d in self.inequalityConstraintNames:
			if id>=d['start'] and id<=d['end']:
				return d
					
	
	def getEqualityConstraintNameFromId(self,id):
		for d in self.equalityConstraintNames:
			if id>=d['start'] and id<=d['end']:
				return d
			
	def findInequalityConstraintsFromName(self,name):
		constraints=[]
		for d in self.inequalityConstraintNames:
			if d['name']==name:
				constraints.append(d)	
		return constraints

	
	def save(self,filename,force_integer=False):
		self.convertToMosek().save(filename,force_integer=force_integer)
		
	def saveMPS(self,filename):
		assert(self.B_lower is None)
		
		f = open(filename, 'w')
		f.write('NAME  exportedFromPython\n')
		f.write('ROWS\n')
		f.write(' N  OBJ\n')
		
		#for i in range(self.Bequalities.size):
			#f.write(' E  E%d\n'%i)
		np.savetxt(f, np.arange(self.Bequalities.size), fmt=' E  E%d',  newline='\n')
			
		#for i in range(self.B_upper.size):
			#f.write(' L  I%d\n'%i)
		np.savetxt(f, np.arange(self.B_upper.size), fmt=' L  I%d',  newline='\n')
				
		f.write('COLUMNS\n')
		
		Aeq=self.Aequalities.tocsc().tocoo() 
		Aineq=self.Ainequalities.tocsc().tocoo()

		
		kEq=0
		kIneq=0
		nEqEntries=len(Aeq.col)
		nIneqEntries=len(Aineq.col)
		
		for i in range(self.nb_variables):
			f.write('    X%-9dOBJ       %f\n'%(i,self.costsvector[i]))

			while kEq<nEqEntries and Aeq.col[kEq]==i:
				f.write('    X%-9dE%-9d%f\n'%(i,Aeq.row[kEq],Aeq.data[kEq]))	
				kEq+=1
			while kIneq<nIneqEntries and Aineq.col[kIneq]==i:
				f.write('    X%-9dI%-9d%f\n'%(i,Aineq.row[kIneq],Aineq.data[kIneq]))	
				kIneq+=1			

		f.write('RHS\n')
		np.savetxt(f,np.column_stack((np.arange(Aeq.shape[0]),self.Bequalities)), fmt='    RHS0      E%-9d%f', newline='\n')
		np.savetxt(f,np.column_stack((np.arange(Aineq.shape[0]),self.B_upper)), fmt='    RHS0      I%-9d%f', newline='\n')
	
			
		f.write('RANGES\n')
		f.write('BOUNDS\n')
		integerIndices=np.nonzero(self.isinteger)[0]
		np.savetxt(f,np.column_stack((integerIndices,self.upperbounds[integerIndices])), fmt=' UI bound     X%-9d%f', newline='\n')
		np.savetxt(f,np.column_stack((integerIndices,self.lowerbounds[integerIndices])), fmt=' LI bound     X%-9d%f', newline='\n')
		continousIndices=np.nonzero(~self.isinteger)[0]
		np.savetxt(f,np.column_stack((continousIndices,self.upperbounds[continousIndices])), fmt=' UP bound     X%-9d%f', newline='\n')
		np.savetxt(f,np.column_stack((continousIndices,self.lowerbounds[continousIndices])), fmt=' LO bound     X%-9d%f', newline='\n')
		
	
		f.write('ENDATA\n')
		f.close()
		
	def save_Ian_E_H_Yen(self,folder):
		if not self.B_lower is None:
			print ('self.B_lower is not None, you should convert your problem with convertToOnesideInequalitySystem first')
			raise
		if not np.all(self.lowerbounds==0):
			print ('lower bound constraint on variables should at 0')
			raise
		
		import os		
		Aeq=self.Aequalities.tocoo()
		tmp=np.row_stack(([Aeq.shape[0],Aeq.shape[1],0.0],np.column_stack((Aeq.row+1,Aeq.col+1,Aeq.data))))
		np.savetxt(os.path.join(folder,'Aeq'),tmp,fmt='%d %d %f')
		np.savetxt(os.path.join(folder,'beq'),self.Bequalities,fmt='%f')
		np.savetxt(os.path.join(folder,'c'),self.costsvector,fmt='%f')
		nbvariables=self.costsvector.size
		upperbounded=np.nonzero(~np.isinf(self.upperbounds))[0]
		nbupperbounded=len(upperbounded)
		Aineq2=sparse.coo_matrix((np.ones(nbupperbounded),(np.arange(nbupperbounded),upperbounded)),(nbupperbounded,nbvariables))
		Aineq=scipy.sparse.vstack((self.Ainequalities,Aineq2)).tocoo()
		bupper=np.hstack((self.B_upper,self.upperbounds[upperbounded]))
		tmp=np.row_stack(([Aineq.shape[0],Aineq.shape[1],0.0],np.column_stack((Aineq.row+1,Aineq.col+1,Aineq.data))))
		np.savetxt(os.path.join(folder,'A'),tmp,fmt='%d %d %f')	
		np.savetxt(os.path.join(folder,'b'),bupper,fmt='%f')	
	
		with open(os.path.join(folder,'meta'), 'w') as f:
			f.write('nb	%d\n'%nbvariables)
			f.write('nf	%d\n'%0)
			f.write('mI	%d\n'%Aineq.shape[0])
			f.write('mE	%d\n'%Aeq.shape[0])
		
	
			
		
	def getVariablesBounds(self):
		types=None
		bl=self.lowerbounds
		bu=self.upperbounds
		
		return types,bl,bu
		
		
	def getVariablesIndices(self,name):
		return  self.variables_dict[name]
		

	def addVariablesArray(self,shape,lowerbounds,upperbounds,costs=0,name=None,isinteger=False):
		if type(shape)==type(0):
			shape=(shape,)
		
		nb_variables_added=np.prod(shape)
		indices=np.arange(nb_variables_added).reshape(shape)+self.nb_variables
		self.nb_variables=self.nb_variables+nb_variables_added
		
		self.Ainequalities._shape=(self.Ainequalities.shape[0],self.nb_variables) 
		self.Aequalities._shape=(self.Aequalities.shape[0],self.nb_variables) 
		
		if type(costs)==type(0) or type(costs)==type(0.0):
			v=costs
			costs=np.empty(shape,dtype=np.float)
			costs.fill(v) 			

		assert np.all(costs.shape==shape)
		lowerbounds,upperbounds=self.convertBoundsToVectors(shape,lowerbounds,upperbounds)
		assert np.all(lowerbounds.shape==shape)
		assert np.all(upperbounds.shape==shape)

		self.upperbounds=np.append(self.upperbounds,upperbounds.ravel())
		self.lowerbounds=np.append(self.lowerbounds,lowerbounds.ravel())
		self.costsvector=np.append(self.costsvector,costs.ravel())
		self.isinteger=np.append(self.isinteger,np.full((nb_variables_added),isinteger,dtype=np.bool))

		if name:
			self.variables_dict[name]=indices
		self.setBoundsOnVariables(indices,lowerbounds,upperbounds)
		return indices

	def convertBoundsToVectors(self,shape,lowerbounds,upperbounds):
		nb_variables=np.prod(np.array(shape))
		if type(lowerbounds)==type(0) or type(lowerbounds)==type(0.0) or type(lowerbounds)==np.float64:
			v=lowerbounds
			lowerbounds=np.empty(shape,dtype=np.float)
			lowerbounds.fill(v) 
		if type(upperbounds)==type(0) or type(upperbounds)==type(0.0) or type(upperbounds)==np.float64:
			v=upperbounds
			upperbounds=np.empty(shape,dtype=np.float)
			upperbounds.fill(v)            

		if upperbounds is None:  
			#assert np.all((lowerbounds.shape==shape))
			upperbounds=np.empty(shape,dtype=np.float)
			upperbounds.fill(np.inf)

		if lowerbounds is None:  
			#assert np.all((upperbounds.shape==shape))
			lowerbounds=np.empty(shape,dtype=np.float)
			lowerbounds.fill(-np.inf)
			
		
		assert np.all((upperbounds.shape==shape))
		assert np.all((lowerbounds.shape==shape)) 			
		
		return lowerbounds,upperbounds

	def setBoundsOnVariables(self,indices,lowerbounds,upperbounds):		
		# could use task.putboundslice if we were sure that the indices is an increasing sequence n with increments of 1 i.e,n+1,n+2,....n+k
		if  (type(lowerbounds)==type(0) or type(lowerbounds)==type(0.0)):
			self.lowerbounds[indices.ravel()]=lowerbounds
		else:	
			self.lowerbounds[indices.ravel()]=lowerbounds.ravel()
		if  (type(upperbounds)==type(0) or type(upperbounds)==type(0.0)):
			self.upperbounds[indices.ravel()]=upperbounds
		else:
			self.upperbounds[indices.ravel()]=upperbounds.ravel()   

	def getVariablesIndices(self,name):
		return  self.variables_dict[name]


	def setCostsVariables(self,indices, costs):
		assert np.all(costs.shape==indices.shape)
		self.costsvector[indices.ravel()]= costs.ravel()
		
	

	def addLinearConstraintRow(self,ids,coefs,lowerbound=None,upperbound=None):
		assert(len(ids)==len(coefs))
		

		if upperbound==lowerbound:  
			csr_matrix_append_row(self.Aequalities,self.nb_variables,ids,coefs)          
			self.Bequalities=np.append(self.Bequalities,lowerbound)              

		else :
			csr_matrix_append_row(self.Ainequalities,self.nb_variables,ids,coefs)            
			if lowerbound==None:
				if not self.B_lower is None:
					self.B_lower=np.append(self.B_lower,-np.inf) 
			else:
				if  self.B_lower is None:
					print ('not coded yet')
				else:
					self.B_lower=np.append(self.B_lower,lowerbound) 
			if upperbound==None:
				if not self.B_upper is None:
					self.B_upper=np.append(self.B_upper,np.inf) 
			else:
				if  self.B_upper is None:
					print ('not coded yet')
				else:
					self.B_upper=np.append(self.B_upper,upperbound) 					
	
			
	def addEqualityConstraintsSparse(self,A,b):
		
		csr_matrix_append_rows(self.Aequalities,A.tocsr())
		self.Bequalities=np.append(self.Bequalities,b) 		

	def addConstraintsSparse(self,A,lowerbounds=None,upperbounds=None):
		# add the constraint lowerbounds<=Ax<=upperbounds to the list of constraints 
		# try to use A as a sparse matrix
		# take advantage of the snipy sparse marices to ease things


		if  (type(lowerbounds)==type(0) or type(lowerbounds)==type(0.0)) and  lowerbounds==upperbounds:
			lowerbounds,upperbounds=self.convertBoundsToVectors((A.shape[0],),lowerbounds,upperbounds)
			csr_matrix_append_rows(self.Aequalities,A.tocsr())
			self.Bequalities=np.append(self.Bequalities,lowerbounds) 
			

		else:
			lowerbounds,upperbounds=self.convertBoundsToVectors((A.shape[0],),lowerbounds,upperbounds)
			csr_matrix_append_rows(self.Ainequalities,A.tocsr())
			self.B_lower=np.append(self.B_lower,lowerbounds) 
			self.B_upper=np.append(self.B_upper,upperbounds)
			
	def addLinearConstraintRows(self,cols,vals,lowerbounds=None,upperbounds=None):
		if not(np.all(np.diff(np.sort(cols,axis=1),axis=1)>0)):
			print('you have twice the same variable in the constraint')
			raise
		#iptr=vals.shape[1]*np.arange(cols.shape[0]+1)
		valsFlat=vals.ravel()
		colsFlat=cols.ravel()
		keep=~(vals==0)
		valsFlat=valsFlat[keep.ravel()]
		colsFlat=colsFlat[keep.ravel()]
		iptr=np.hstack(([0],np.cumsum(np.sum(keep,axis=1))))
		A=scipy.sparse.csr_matrix((valsFlat,colsFlat,iptr))				
		self.addConstraintsSparse(A,lowerbounds=lowerbounds,upperbounds=upperbounds)	

	def addSoftLinearConstraintRows(self,cols,vals,lowerbounds=None,upperbounds=None,coef_penalization=0):
		if np.all(coef_penalization==np.inf):
			self.addLinearConstraintRows(cols,vals,lowerbounds=lowerbounds,upperbounds=upperbounds)

		else:
			aux= self.addVariablesArray((cols.shape[0],),upperbounds=None,lowerbounds=0,costs=coef_penalization)

			cols2=np.column_stack((cols,aux))
			if not upperbounds is None:
				vals2=np.column_stack((vals,-np.ones((vals.shape[0],1)))	)	
				self.addLinearConstraintRows(cols2,vals2,lowerbounds=None,upperbounds=upperbounds)
			if not lowerbounds is None:	
				vals2=np.column_stack((vals,np.ones((vals.shape[0],1))))		
				self.addLinearConstraintRows(cols2,vals2,lowerbounds,upperbounds=None)			
			return aux

	def addLinearConstraintsWithBroadcasting(self,cols, vals,lowerbounds=None,upperbounds=None):
		cols2=cols.reshape(-1,cols.shape[-1])
		vals2=np.tile(np.array(vals), (cols2.shape[0],1))
		self.addLinearConstraintRows(cols2, vals2,lowerbounds=lowerbounds,upperbounds=upperbounds)

	def addInequalities(self,indicesAndWeightPairs,lowerbounds,upperbounds,check=True):
		cols=[]
		vals=[]
		for t in indicesAndWeightPairs:
			cols.append(t[0].flatten())
			vals.append((np.ones(t[0].shape)*t[1]).flatten())# trick to do broadcasting on vals if needed
		if isinstance(upperbounds,np.ndarray):
			upperbounds=upperbounds.flatten()
		self.addLinearConstraintRows(np.column_stack(cols),np.column_stack(vals),lowerbounds,upperbounds)
		if (not self.solution is None) and check:
			assert(self.checkSolution(self.solution))



	def removeFixedVariables(self):
		# should you more complete presolve procedure in case we use interior point method
		# http://www.davi.ws/doc/gondzio94presolve.pdf
		free=self.upperbounds>self.lowerbounds
		idfree=np.nonzero(free)[0]
		nbfree=np.sum(free)
		Mchange=scipy.sparse.coo_matrix((np.ones((nbfree)),(idfree,np.arange(nbfree))),(self.nb_variables,nbfree))
		shift=np.zeros((self.nb_variables))
		shift[~free]=self.lowerbounds[~free]
		
		self.Bequalities= self.Bequalities-self.Aequalities*shift
		if not self.B_lower is None:
			self.B_lower=self.B_lower-self.Ainequalities*shift
		if not self.B_upper is None:
			self.B_upper=self.B_upper-self.Ainequalities*shift
		
		#self.Bequalities equalities= self.Bequalities-self.Aequalities*shift
		
		self.costsvector=self.costsvector[free]
		b=self.Ainequalities.__dict__['blocks']
		self.Ainequalities=self.Ainequalities[:,free]
		self.Ainequalities.__dict__['blocks']=b
		
		#u,ia,ib=unique_rows(self.Ainequalities.todense())
		
		
		b=self.Aequalities.__dict__['blocks']
		self.Aequalities=self.Aequalities[:,free]
		self.Aequalities.__dict__['blocks']=b
		
		#find constrinats with single variable left
		#self.Aequalities
		
		#u,ia,ib=unique_rows(self.Aequalities.todense())
		self.nb_variables=nbfree
		self.lowerbounds=self.lowerbounds[free]
		self.upperbounds=self.upperbounds[free]
		
		#Bequalities= LP.Bequalities
		#Bequalities= LP.Bequalities
		return Mchange,shift

	
	def convertToSlackForm(self):
		"""convert to the form min_y c.t Ay=b y>=0 by adding slack variables and shift on x
		the solution of the original problem is obtained using x = Mchange*y+ shift with 
		y the solution of the new problem
		have a look at https://ocw.mit.edu/courses/sloan-school-of-management/15-053-optimization-methods-in-management-science-spring-2013/tutorials/MIT15_053S13_tut06.pdf
		"""
		
		self.convertToOnesideInequalitySystem()
			
		# inverse variables that are only bounded above using a change of variable x=M*y
		reverse=np.isinf(self.lowerbounds)&(~np.isinf(self.upperbounds))
		if np.any(reverse):
			raise ('this part of the code has not been tested yet')
			d=np.ones(self.nb_variables)
			d[reverse]=-1
			M1=scipy.sparse.spdiags([d],[0],self.nb_variables,self.nb_variables)
			Ainequalities=None
			Aequalities=None
			if not self.Ainequalities is None:
				Ainequalities = self.Ainequalities*M1
				Ainequalities.__dict__['blocks']=[(0,Ainequalities.shape[0]-1)]
			if not self.Aequalities is None:	
				Aequalities   = self.Aequalities*M1
				Aequalities.__dict__['blocks']=[(0,Aequalities.shape[0]-1)]
			lowerbounds=copy.copy(self.lowerbounds)
			upperbounds=copy.copy(self.upperbounds)
			lowerbounds[reverse]=-self.upperbounds[reverse]
			upperbounds[reverse]=-self.lowerbounds[reverse]
			
		else:
			M1=scipy.sparse.eye(self.nb_variables)
			lowerbounds=copy.copy(self.lowerbounds)
			upperbounds=copy.copy(self.upperbounds)	
			Ainequalities = copy.copy(self.Ainequalities)
			Aequalities = copy.copy(self.Aequalities)
			
		
		# shift lower bounds to 0 by a change of variable y =x-lb 
		#Ax=b lb<=x<=ub =>	Ay=A(x-lb)=b-A*lb
		shift=np.zeros(lowerbounds.size)
		shift[~np.isinf(lowerbounds)]=lowerbounds[~np.isinf(lowerbounds)]
		assert(self.B_lower is None)
		B_upper=self.B_upper-Ainequalities*shift	
		
		if not self.Bequalities is None:
			Bequalities=self.Bequalities-Aequalities*shift
		else:
			Bequalities=None
		
		upperbounds=upperbounds-shift
		lowerbounds=lowerbounds-shift
		
		#put upper bound constraints into the inequality matrix
		idupper=np.nonzero(~np.isinf(self.upperbounds))[0]
		nbupper=len(idupper)
		if nbupper>0:
			#raise 'this part of the code has not been tested yet'
			T=scipy.sparse.coo_matrix((np.ones(nbupper),(np.arange(nbupper),idupper)))
			csr_matrix_append_rows(Ainequalities,T.tocsr())
			B_upper=np.append(B_upper,upperbounds[idupper])
		upperbounds=None
		
		# replace free variables by a difference of positive variables
		free=np.isinf(-self.lowerbounds)& np.isinf(self.upperbounds)
		# create the permutation matrix that set the all free variables after the other variables 
		nbfree=np.sum(free)
		nb_variables=self.nb_variables
		costsvector=self.costsvector
		if nbfree>0:
			#raise 'this part of the code has not been tested yet'
			nbnotfree=nb_variables-nbfree
			J=(np.cumsum(~free)-1)*(~free)+(np.cumsum(free)+nbnotfree-1)*(free)
			perm=scipy.sparse.coo_matrix((np.ones(self.nb_variables),(np.arange(self.nb_variables),J)))
			tmp=scipy.sparse.vstack((scipy.sparse.hstack((scipy.sparse.eye(nbnotfree),scipy.sparse.coo_matrix((nbnotfree,2*nbfree)))),
		                         scipy.sparse.hstack((scipy.sparse.coo_matrix((nbfree,nbnotfree)),scipy.sparse.eye(nbfree),-scipy.sparse.eye(nbfree)))))			
			M2=perm*tmp
			Mchange=M1*M2
			nb_variables=nbnotfree+2*nbfree
			lowerbounds=np.zeros(nb_variables)
			costsvector=costsvector*Mchange
			if not Aequalities is None:
				Aequalities=Aequalities*Mchange
				Aequalities.__dict__['blocks']=[(0,Aequalities.shape[0]-1)]
			if not Ainequalities is None:	
				Ainequalities=Ainequalities*Mchange
				Ainequalities.__dict__['blocks']=[(0,Ainequalities.shape[0]-1)]
		else:
			Mchange=M1
			
		# remove lower inequality constraints 
		
		# replace inequality constraint Ax<=b  by ax+s=b s>=0
		nbslack=Ainequalities.shape[0]
		
		
		
		nb_variables=nb_variables+nbslack
		Ainequalities=scipy.sparse.hstack((Ainequalities,scipy.sparse.eye(nbslack)))
		Aequalities._shape=(Aequalities.shape[0],nb_variables) 
		Mchange=Mchange.tocsr()
		Mchange._shape=(Mchange.shape[0],nb_variables) 

		lowerbounds=np.append(lowerbounds,np.zeros(nbslack))
		costsvector=np.append(costsvector,np.zeros(nbslack))		
		
		
		csr_matrix_append_rows(Aequalities,Ainequalities.tocsr())
		Bequalities=np.append(Bequalities,B_upper)
					
		
		B_lower=None
		B_upper=None
		Ainequalities=None
		
		self.nb_variables=nb_variables
		self.B_lower=B_lower
		self.B_upper=B_upper
		self.Ainequalities=Ainequalities
		self.Bequalities=Bequalities
		self.Aequalities=Aequalities
		self.lowerbounds=lowerbounds
		self.upperbounds=upperbounds
		self.costsvector=costsvector
			
			
		return Mchange,shift

		
	def convertToAllEqualities(self):
		"""convert to the form min c.t Ax=b lb<=x<=ub by adding slack variables
		the solution of th original problem is obtained using the first elements in x"""
		if not self.Ainequalities is None:			
			m=self.Ainequalities.shape[0]
			n=self.Ainequalities.shape[1]
			slacks_lower=self.addVariablesArray(m, self.B_lower, self.B_upper)
			self.Ainequalities._shape=(self.Ainequalities.shape[0],n) 
			self.addConstraintsSparse(scipy.sparse.hstack((self.Ainequalities,-scipy.sparse.eye(m))),0,0)
			self.B_lower=None
			self.B_upper=None
			self.Ainequalities=None
		
		
		
	def convertToOnesideInequalitySystem(self):
		"""convert to the form min c.t Aineq x<=b_ineq Ax=b lb<=x<=ub by adding augmenting the size of Aineq"""		
		if (not self.Ainequalities is None) and (not self.B_lower is None):
			idskeep_upper=np.nonzero(self.B_upper!=np.inf)[0]
			mapping_upper=np.hstack(([0],np.cumsum(self.B_upper!=np.inf)))
			idskeep_lower=np.nonzero(self.B_lower!=-np.inf)[0]
			mapping_lower=np.hstack(([0],np.cumsum(self.B_lower!=np.inf)))
			if len(idskeep_lower)>0 and len(idskeep_upper)>0:
				
				newInequalityConstraintNames=[]
				for d in self.inequalityConstraintNames:
					d={'name':d['name'],\
					'start':mapping_upper[d['start']],\
					'end':mapping_upper[d['end']]}
					newInequalityConstraintNames.append(d)
				for d in self.inequalityConstraintNames:
					
					d={'name':d['name'],\
					   'start':idskeep_upper.size+mapping_lower[d['start']],\
					   'end':idskeep_upper.size+mapping_lower[d['end']]}
					newInequalityConstraintNames.append(d)	
					
				self.inequalityConstraintNames=newInequalityConstraintNames			
				self.Ainequalities=scipy.sparse.vstack((self.Ainequalities[idskeep_upper,:],\
					                                -self.Ainequalities[idskeep_lower,:])).tocsr()
			
				
	
				
			elif len(idskeep_lower)>0 :
				self.Ainequalities=-self.Ainequalities
			else:
				self.Ainequalities=self.Ainequalities
			self.Ainequalities.__dict__['blocks']=[(0,self.Ainequalities.shape[0]-1)]
			self.B_upper=np.hstack((self.B_upper[idskeep_upper],-self.B_lower[idskeep_lower]))
			self.B_lower=None
		
	def convertToAllInequalities(self):
		"""convert to the form min c.t b_lower<=Aineq x<=b_upper lb<=x<=ub by adding augmenting the size of Aineq"""
		if not self.Aequalities is None:
			
			newInequalityConstraintNames=[]
			for d in self.equalityConstraintNames:
				newInequalityConstraintNames.append(d)
			for d in self.inequalityConstraintNames:
		
				d={'name':d['name'],\
				   'start':self.Aequalities.shape[0]+d['start'],\
				   'end':self.Aequalities.shape[0]+d['end']}
				newInequalityConstraintNames.append(d)	
			self.inequalityConstraintNames=newInequalityConstraintNames
			self.equalityConstraintNames=[]
			
			self.Ainequalities=scipy.sparse.vstack((self.Aequalities,self.Ainequalities))
			if self.B_lower is None:
				self.B_lower=np.full((self.Ainequalities.shape[0]),-np.inf)
			self.B_lower=np.hstack((self.Bequalities,self.B_lower))
			if self.B_upper is None:
				self.B_upper=np.full((self.Ainequalities.shape[0]),np.inf)		
			self.B_upper=np.hstack((self.Bequalities,self.B_upper))	
			self.Aequalities=None
			self.Bequalities=None
			
			
	def convertToCVXPY(self):
		
		
		if not(self.Ainequalities is None) and self.Ainequalities.shape[0]>0:
			check_csr_matrix(self.Ainequalities)
			Aineq=self.Ainequalities
		else:
			Aineq=None
		if self.Aequalities.shape[0]>0:
			Aeq=self.Aequalities
			Beq=self.Bequalities
		else:
			Aeq=None
			Beq=None
		
		#uses cvxpy to call SCS as it make its easier to specify tje problem
		# Problem data.
		# Construct the problem.
		x = cvxpy.Variable(self.nb_variables)
		objective = cvxpy.Minimize(np.matrix(self.costsvector[None,:])*x)
		
		constraints =[]
		
		if np.all(np.isinf(self.lowerbounds)):
			pass
		elif np.any(np.isinf(self.lowerbounds)):
			print ('not code yet')
			raise
		else:
			constraints.append( self.lowerbounds <= x )	
			
		if np.all(np.isinf(self.upperbounds)):
			pass
		elif np.any(np.isinf(self.upperbounds)):
			print ('not code yet')
			raise
		else:
			constraints.append( x <= self.upperbounds)
		
		if not Aineq is None:
			if not self.B_upper is None:
				if np.all(np.isinf(self.B_upper)):
					pass	
				elif np.any(np.isinf(self.B_upper)):
					print ('not yet coded')
					raise
				else:
					constraints.append(Aineq*x<=self.B_upper)
			if not self.B_lower is None:
				if np.all(np.isinf(self.B_lower)):
					pass					
				elif np.any(np.isinf(self.B_lower)):
					print( 'not yet coded')
					raise	
				else:
					constraints.append(self.B_lower<=Aineq*x)
		if not  Aeq is None:
			constraints.append(Aeq*x==self.Bequalities)
		prob = cvxpy.Problem(objective, constraints)	
		return prob,x
			
	def solve(self,method="ADMM",getTiming=True,\
	          x0=None,nb_iter=10000,\
	          max_time=None,callbackFunc=None,\
	          nb_iter_plot=10,\
	          plotSolution=None,groundTruth=None,groundTruthIndices=None):
		
		
		if not(self.Ainequalities is None) and self.Ainequalities.shape[0]>0:
			check_csr_matrix(self.Ainequalities)
			Aineq=self.Ainequalities
		else:
			Aineq=None
		if self.Aequalities.shape[0]>0:
			Aeq=self.Aequalities
			Beq=self.Bequalities
		else:
			Aeq=None
			Beq=None
			
			
		start = time.clock() 	
		
		self.distanceToGroundTruth=[]	
		self.distanceToGroundTruthAfterRounding=[]	
		self.opttime_curve=[]
		self.dopttime_curve=[]
		self.pobj_curve=[]
		self.dobj_curve=[]
		self.pobjbound=[]
		self.max_violated_inequality=[]
		self.max_violated_equality=[]
		self.max_violated_constraint=[]
		self.itrn_curve=[]
		
		
		
		def simplexCallBack(solution,**kwargs):								
			if not groundTruth is None:
					self.distanceToGroundTruth.append(np.mean(np.abs(groundTruth-solution[groundTruthIndices])))
					self.distanceToGroundTruthAfterRounding.append(np.mean(np.abs(groundTruth-np.round(solution[groundTruthIndices]))))
			duration=time.clock() -start
			self.opttime_curve.append(duration)
			self.pobj_curve.append(self.costsvector.dot(solution.T))
			maxv=self.maxConstraintViolation(solution)
			self.max_violated_constraint.append(maxv)
			
		
		
		def callbackFunc(niter,solution,energy1,energy2,duration,max_violated_equality,max_violated_inequality,is_active_variable=None):
			if not groundTruth is None:
				self.distanceToGroundTruth.append(np.mean(np.abs(groundTruth-solution[groundTruthIndices])))
				self.distanceToGroundTruthAfterRounding.append(np.mean(np.abs(groundTruth-np.round(solution[groundTruthIndices]))))
			self.itrn_curve.append(niter)
			self.opttime_curve.append(duration)
			self.dopttime_curve.append(duration)
			self.dobj_curve.append(energy2)
			self.pobj_curve.append(energy1)
			maxv=self.maxConstraintViolation(solution)
			self.max_violated_constraint.append(maxv)
			self.max_violated_equality.append(max_violated_equality)
			self.max_violated_inequality.append(max_violated_inequality)
			if not plotSolution is None:
				plotSolution(niter,solution,is_active_variable=is_active_variable)
				
		
		if not method in solving_methods:
			print ('method %s not valid'%method)
			print ('avalaible method are')
			for vmethod in solving_methods:
				print (vmethod)
			raise
		if method=='ScipyLinProg':
			if not (self.B_lower is None):
				print ('you need to convert your lp to a one side inequality system using convertToOnesideInequalitySystem')
				raise
			if Aeq is None:
				A_eq=None
				b_eq=None
			else:
				A_eq=Aeq.toarray()
				b_eq=Beq
			sol=scipy.optimize.linprog(self.costsvector, A_ub=Aineq.toarray(), b_ub=self.B_upper, A_eq=A_eq, b_eq=b_eq, 
			                        bounds=np.column_stack((self.lowerbounds,self.upperbounds)), 
			                        method='simplex',
			                        callback=simplexCallBack)
			#if not sol['success']:
				#raise BaseException(sol['message'])
			x=sol['x']
			
		elif method== 'Mehrotra':
			
			LPslack=copy.deepcopy(self)
			Mchange1,shift1=LPslack.removeFixedVariables()# removed fixed variables
			Mchange2,shift2=LPslack.convertToSlackForm()
			def MehrotraCallBack(solution,niter,**kwargs):
				x=Mchange2*solution-shift2
				x=Mchange1*x-shift1
				self.itrn_curve.append(niter)
				simplexCallBack(x)
				
			f, x, y, s, N=mpcSol(LPslack.Aequalities, LPslack.Bequalities, LPslack.costsvector, callBack=MehrotraCallBack)
			LPslack.checkSolution(x)
			x=Mchange2*x-shift2
			x=Mchange1*x-shift1
			self.checkSolution(x)
			self.maxConstraintViolation(x)
			
			
		elif method =='CVXOPT':
			
			prob,x=self.convertToCVXPY()
			# The optimal objective is returned by prob.solve().
			result = prob.solve(verbose=True,solver=cvxpy.CVXOPT)		
			x=np.array(x.value).flatten()
			
			#from cvxopt import matrix, solvers
			## the format axcpt
			#LPslack=copy.deepcopy(self)
			#Mchange,shift=LPslack.convertToSlackForm()
			##sol=solvers.lp(c, G, h, A , b)  # uses cvxopt conic solver
			## need to rewrite the problem in the form
			## minimize    c'*x
			## subject to  G*x <= h and A*x = b
			#print 'not yet coded'
			#raise
		
		elif method =='SCS':
			prob,x=self.convertToCVXPY()
			# The optimal objective is returned by prob.solve().
			result = prob.solve(verbose=True,solver=cvxpy.SCS,max_iters=10000,eps=1e-5)		
			x=np.array(x.value).flatten()
			
		elif method =='ECOS':
			prob,x=self.convertToCVXPY()
			# The optimal objective is returned by prob.solve().
			result = prob.solve(verbose=True,solver=cvxpy.ECOS)		
			x=np.array(x.value).flatten()					
		
			
			
			
		elif method=='ADMM':		
			x=LP_admm(self.costsvector,Aeq,Beq,\
				Aineq,self.B_lower,self.B_upper,\
				self.lowerbounds,self.upperbounds,nb_iter=nb_iter,\
			        x0=x0,callbackFunc=callbackFunc,max_time=max_time,nb_iter_plot=nb_iter_plot)
			
		elif method=='ADMMBlocks':		
			x=LP_admmBlockDecomposition(self.costsvector,Aeq,Beq,\
		                  Aineq,self.B_lower,self.B_upper,\
		                  self.lowerbounds,self.upperbounds,nb_iter=nb_iter,\
			          nb_iter_plot=nb_iter_plot,\
		                  x0=x0,callbackFunc=callbackFunc,max_time=max_time)					
		elif method=='ADMM2':		
			x=LP_admm2(self.costsvector,Aeq,Beq,\
		                  Aineq,self.B_lower,self.B_upper,\
		                  self.lowerbounds,self.upperbounds,nb_iter=nb_iter,\
		                  x0=x0,callbackFunc=callbackFunc,max_time=max_time,nb_iter_plot=nb_iter_plot	)	

		elif method=="ChambollePockPPD"	:	
			LPreduced=copy.deepcopy(self)
			Mchange1,shift1=LPreduced.removeFixedVariables()# removed fixed variables
			def thisBack(niter,solution,energy1,energy2,duration,max_violated_equality,max_violated_inequality):
				solution=Mchange1*solution-shift1
				callbackFunc(niter, solution,energy1,energy2,duration,max_violated_equality,max_violated_inequality )
			
			x,best_integer_solution=ChambollePockPPD(LPreduced.costsvector,LPreduced.Aequalities,LPreduced.Bequalities,\
			                                        LPreduced.Ainequalities,LPreduced.B_lower,LPreduced.B_upper,\
			                   LPreduced.lowerbounds,LPreduced.upperbounds,\
			                   	x0=None,alpha=1,theta=1,nb_iter=nb_iter,callbackFunc=thisBack,max_time=max_time,save_problem=False,nb_iter_plot=nb_iter_plot)
			x=Mchange1*x-shift1	
		elif method=="ChambollePockPPDAS"	:	
			LPreduced=copy.deepcopy(self)
			Mchange1,shift1=LPreduced.removeFixedVariables()# removed fixed variables
			def thisBack(niter,solution,energy1,energy2,duration,max_violated_equality,max_violated_inequality,is_active_variable):
				solution=Mchange1*solution-shift1
				callbackFunc(niter, solution,energy1,energy2,duration,max_violated_equality,max_violated_inequality )
						
			
			x=ChambollePockPPDAS(LPreduced,\
					     x0=x0,alpha=1,theta=1,nb_iter=nb_iter,\
					     nb_iter_plot=nb_iter_plot,\
					     frequency_update_active_set=20,
					     callbackFunc=thisBack,max_time=max_time)			
			x=Mchange1*x-shift1			
		
			
		elif method=="DualGradientAscent":
			x, y_eq,y_ineq=DualGradientAscent(x=x0,LP=self,nbmaxiter=nb_iter,callbackFunc=callbackFunc,y_eq=None,y_ineq=None,max_time=max_time,nb_iter_plot=nb_iter_plot)
		elif method=="DualCoordinateAscent":
			LPreduced=copy.deepcopy(self)
			Mchange1,shift1=LPreduced.removeFixedVariables()# removed fixed variables
			def thisBack(niter,solution,energy1,energy2,duration,max_violated_equality,max_violated_inequality):
				solution=Mchange1*solution-shift1
				callbackFunc(niter, solution,energy1,energy2,duration,max_violated_equality,max_violated_inequality )
						
			x, y_eq,y_ineq=DualCoordinateAscent(x=None,LP=LPreduced,nbmaxiter=nb_iter,callbackFunc=thisBack,y_eq=None,y_ineq=None,max_time=max_time,nb_iter_plot=nb_iter_plot)
			x=Mchange1*x-shift1

		else: 
			print ("unkown LP solver method "+method)
			raise
		elapsed= (time.clock() - start)		
		
		if getTiming:
			
			return x,elapsed
		else:
			return x
		

