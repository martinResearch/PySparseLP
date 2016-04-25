import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse
import scipy.ndimage
import sys
from ADMM import *
from ChambollePockPreconditionedPrimalDual import *
from ADMMBlocks import *
from DualGradientAscent import *
from DualCoordinateAscent import *

def csr_matrix_append_row(A,n,cols,vals):
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

class SparseLP():
	def __init__(self):
		# start writing the linear program 

		
		self.nb_variables=0
		self.variables_dict=dict()        
		self.upperbounds=np.empty((0),dtype=np.float)
		self.lowerbounds=np.empty((0),dtype=np.float)
		self.costsvector=np.empty((0),dtype=np.float)
		self.Ainequalities=empty_csr_matrix()
		self.Ainequalities.__dict__['blocks']=[]
		self.B_lower=np.empty((0),dtype=np.float)
		self.B_upper=np.empty((0),dtype=np.float)
		self.Aequalities=empty_csr_matrix()
		self.Bequalities=np.empty((0),dtype=np.float)   
		self.Aequalities.__dict__['blocks']=[]
		self.solver='chambolle_pock'
		self.simplex=None
		self.equalityConstraintNames=[]
		self.inequalityConstraintNames=[]
		

	def checkSolution(self,solution,tol=1e-6):
		types, lb, ub=self.getVariablesBounds()
		valid=True
		valid=valid&(np.max(lb-solution)<tol)
		valid=valid&(np.max(solution-ub)<tol)
		if self.Aequalities.shape[0]>0:
			valid=valid&(np.max(np.abs(self.Aequalities*solution-self.Bequalities))<tol)
		if self.Ainequalities.shape[0]>0:
			valid=valid&(np.max(self.Ainequalities*solution-self.B_upper)<tol)
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

	def convertToMosek(self):
		import mosekHelper
		moseklp=mosekHelper.MosekHelper()
		print 'conversion to mosek LP : adding variables'
		moseklp.addVariablesArray((self.nb_variables),self.lowerbounds,self.upperbounds,name=None,costs=self.costsvector)
		print  'conversion to mosek LP : adding inequalities'
		if not(self.Ainequalities is None):
			moseklp.addConstraintsSparse(self.Ainequalities,lowerbounds=self.B_lower,upperbounds=self.B_upper)
		print  'conversion to mosek LP : adding equalities'
		moseklp.addConstraintsSparse(self.Aequalities,lowerbounds=self.Bequalities,upperbounds=self.Bequalities)
		return moseklp
	
	def save(self,filename,force_integer=False):
		self.convertToMosek().save(filename,force_integer=force_integer)
		
	def saveMPS(self,filename):
		assert(self.B_lower is None)
		self.isinteger=np.ones(self.costsvector.size,dtype=bool)
		f = open(filename, 'w')
		f.write('NAME  exportedFromPython\n')
		f.write('ROWS\n')
		f.write(' N  OBJ\n')
		
		for i in range(self.Bequalities.size):
			f.write(' E  E%d\n',i)
			
			
		for i in range(self.B_upper.size):
			f.write(' L  I%d\n'%i)
				
		f.write('COLUMNS\n')
		Aeq=self.Aequalities.tocsc()
		Aineq=self.Ainequalities.tocsc()
		for i in range(self.nb_variables):
			f.write('    X%-9dOBJ       %f\n'%(i,self.costsvector[i]))
			col=Aeq[:,i]
			for j,d in zip(col.indices,col.data):
				f.write('    X%-9dE%-9d%f\n'%(i,j,d))
	
			col=Aineq[:,i]				
			for j,d in zip(col.indices,col.data):
				f.write('    X%-9dI%-9d%f\n'%(i,j,d))
				
		#for i in range(self.costsvector.size):
			#f.write('    X%-9dOBJ       %f\n'%(i,self.costsvector[i]))
			
			
		#Aeq=self.Aequalities.tocsc().tocoo()# convert to scipy sparse csc matrix
		#assert np.all((np.diff(Aeq.col)>=0))
		#for i in range(Aeq.data.size):
			#f.write('    X%-9dEQ%-8d%f\n'%(Aeq.col[i],Aeq.row[i]+1,Aeq.data[i]))		
		
		#Aineq=self.Ainequalities.tocsc().tocoo()# convert to scipy sparse csc matrix
		#assert np.all((np.diff(Aineq.col)>=0))
		#for i in range(Aineq.data.size):
			#f.write('    X%-9dINEQ%-6d%f\n'%(Aineq.col[i],Aineq.row[i],Aineq.data[i]))
		
		f.write('RHS\n')
		for i in range(Aeq.shape[0]):
			f.write('    RHS0      E%-9d%f\n'%(i,self.Bequalities[i]))
		for i in range(Aineq.shape[0]):
			f.write('    RHS0      I%-9d%f\n'%(i,self.B_upper[i]))	
			
		f.write('RANGES\n')
		f.write('BOUNDS\n')
		for i in range(self.upperbounds.size):
			if self.isinteger[i]:
				f.write(' UI bound     X%-9d%f\n'%(i,self.upperbounds[i]))
			else:
				f.write(' UP bound     X%-9d%f\n'%(i,self.upperbounds[i]))
		for i in range(self.upperbounds.size ):
			if self.isinteger[i]:
				f.write(' LI bound     X%-9d%f\n'%(i,self.lowerbounds[i]))
			else:
				f.write(' LO bound     X%-9d%f\n'%(i,self.lowerbounds[i]))
		f.write('ENDATA\n')
		f.close()
		
			
		
	def getVariablesBounds(self):
		types=None
		bl=self.lowerbounds
		bu=self.upperbounds
		
		return types,bl,bu
		
		
	def getVariablesIndices(self,name):
		return  self.variables_dict[name]
		

	def addVariablesArray(self,shape,lowerbounds,upperbounds,costs=0,name=None):
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

		if name:
			self.variables_dict[name]=indices
		self.setBoundsOnVariables(indices,lowerbounds,upperbounds)
		return indices

	def convertBoundsToVectors(self,shape,lowerbounds,upperbounds):
		nb_variables=np.prod(np.array(shape))
		if type(lowerbounds)==type(0) or type(lowerbounds)==type(0.0):
			v=lowerbounds
			lowerbounds=np.empty(shape,dtype=np.float)
			lowerbounds.fill(v) 
		if type(upperbounds)==type(0) or type(upperbounds)==type(0.0):
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

		self.lowerbounds[indices.ravel()]=lowerbounds
		self.upperbounds[indices.ravel()]=upperbounds   

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
				lowerbound=-np.inf
			if upperbound==None:
				upperbound=-np.inf         
			self.B_lower=np.append(self.B_lower,lowerbound)   
			self.B_upper=np.append(self.B_upper,upperbound)   


	def addConstraintsSparse(self,A,lowerbounds=None,upperbounds=None):
		# add the constraint lowerbounds<=Ax<=upperbounds to the list of constraints 
		# try to use A as a sparse matrix
		# take advantage of the snipy sparse marices to ease things


		if lowerbounds==upperbounds:
			lowerbounds,upperbounds=self.convertBoundsToVectors((A.shape[0],),lowerbounds,upperbounds)
			csr_matrix_append_rows(self.Aequalities,A.tocsr())
			self.Bequalities=np.append(self.Bequalities,lowerbounds) 
			

		else:
			lowerbounds,upperbounds=self.convertBoundsToVectors((A.shape[0],),lowerbounds,upperbounds)
			csr_matrix_append_rows(self.Ainequalities,A.tocsr())
			self.B_lower=np.append(self.B_lower,lowerbounds) 
			self.B_upper=np.append(self.B_upper,upperbounds)
			
	def addLinearConstraintRows(self,cols,vals,lowerbounds=None,upperbounds=None):
		assert(np.all(np.diff(np.sort(cols,axis=1),axis=1)>0))
		iptr=vals.shape[1]*np.arange(cols.shape[0]+1)
		A=scipy.sparse.csr_matrix((vals.ravel(),cols.ravel(),iptr))				
		self.addConstraintsSparse(A,lowerbounds=lowerbounds,upperbounds=upperbounds)	

	def addSoftLinearConstraintRows(self,cols,vals,lowerbounds=None,upperbounds=None,coef_penalization=0):
		if np.all(coef_penalization==np.inf):
			self.addLinearConstraintRows(cols,vals,lowerbounds=lowerbounds,upperbounds=upperbounds)

		else:
			aux= self.addVariablesArray((cols.shape[0],),upperbounds=None,lowerbounds=0,costs=coef_penalization)

			cols2=np.column_stack((cols,aux))
			if upperbounds!=None:
				vals2=np.column_stack((vals,-np.ones((vals.shape[0],1)))	)	
				self.addLinearConstraintRows(cols2,vals2,lowerbounds=None,upperbounds=upperbounds)
			if lowerbounds!=None:	
				vals2=np.column_stack((vals,np.ones((vals.shape[0],1))))		
				self.addLinearConstraintRows(cols2,vals2,lowerbounds,upperbounds=None)			
			return aux

	def addLinearConstraintsWithBroadcasting(self,cols, vals,lowerbounds=None,upperbounds=None):
		cols2=cols.reshape(-1,cols.shape[-1])
		vals2=np.tile(np.array(vals), (cols2.shape[0],1))
		self.addLinearConstraintRows(cols2, vals2,lowerbounds=lowerbounds,upperbounds=upperbounds)

			
	def addSimplex(self,indices):
		if not self.simplex is None:
			print "can define a simplex only once"
			raise
		self.simplex=indices
		
	def convertToSlackForm(self):
		if self.Ainequalities!=None:			
			m=self.Ainequalities.shape[0]
			n=self.Ainequalities.shape[1]
			slacks_lower=self.addVariablesArray(m, self.B_lower, self.B_upper)
			self.Ainequalities._shape=(self.Ainequalities.shape[0],n) 
			self.addConstraintsSparse(scipy.sparse.hstack((self.Ainequalities,-scipy.sparse.eye(m))),0,0)
			self.B_lower=None
			self.B_lupper=None
			self.Ainequalities=None
		
		
		
	def convertToOnesideInequalitySystem(self):
		if self.Ainequalities!=None and (not self.B_lower is None):
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
			self.B_upper=np.hstack((self.B_upper[idskeep_upper],-self.B_lower[idskeep_lower]))
			self.B_lower=None
		
	def convertToAllInequalities(self):	
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
			self.B_lower=np.hstack((self.Bequalities,self.B_lower))
			self.B_upper=np.hstack((self.Bequalities,self.B_upper))	
			self.Aequalities=None
			self.Bequalities=None
			
			
		

			
	def solve(self,method="ADMM",force_integer=False,getTiming=True,\
	          x0=None,nb_iter=10000,\
	          max_time=None,callbackFunc=None,\
	          nb_iter_plot=10,\
	          plotSolution=None,groundTruth=None,groundTruthIndices=None,frequency_update_active_set=2000000):
		
		
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
		self.itrn_curve=[]
		
		def callbackFunc(niter,solution,energy1,energy2,duration,max_violated_equality,max_violated_inequality,is_active_variable=None):
			if groundTruth!=None:
				self.distanceToGroundTruth.append(np.mean(np.abs(groundTruth-solution[groundTruthIndices])))
				self.distanceToGroundTruthAfterRounding.append(np.mean(np.abs(groundTruth-np.round(solution[groundTruthIndices]))))
			self.itrn_curve.append(niter)
			self.opttime_curve.append(duration)
			self.dopttime_curve.append(duration)
			self.dobj_curve.append(energy2)
			self.pobj_curve.append(energy1)
			self.max_violated_equality.append(max_violated_equality)
			self.max_violated_inequality.append(max_violated_inequality)
			if plotSolution!=None:
				plotSolution(niter,solution,is_active_variable=is_active_variable)
				
		
				
		if method=='ScipyLinProg':
			if not (self.B_lower is None):
				print 'you need to convert your lp to a one side inequality system using convertToOnesideInequalitySystem'
				raise
			if Aeq is None:
				A_eq=None
				b_eq=None
			else:
				A_eq=Aeq.toarray()
				b_eq=Beq
			sol=scipy.optimize.linprog(self.costsvector, A_ub=Aineq.toarray(), b_ub=self.B_upper, A_eq=A_eq, b_eq=b_eq, 
			                        bounds=np.column_stack((self.lowerbounds,self.upperbounds)), 
			                        method='simplex')
			if not sol['success']:
				raise
			x=sol['x']
		elif method=='ADMM':		
			x=LP_admm(self.costsvector,Aeq,Beq,\
				Aineq,self.B_lower,self.B_upper,\
				self.lowerbounds,self.upperbounds,nb_iter=nb_iter,\
			        x0=x0,callbackFunc=callbackFunc,max_time=max_time)
			
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
		                  x0=x0,callbackFunc=callbackFunc,max_time=max_time)		
		#x=LP_reprojected(self.costsvector,Aeq,Beq,\
		                #Aineq,Bineq,\
		                #self.lowerbounds,self.upperbounds,\
		                #x0=x0,callbackFunc=callbackFunc,max_time=max_time)
		elif method=="ChambollePock"	:	
			x=LP_primalDual(self,\
			      simplex=self.simplex,
		              x0=x0,alpha=1,theta=1,nb_iter=nb_iter,\
			      nb_iter_plot=nb_iter_plot,\
			      frequency_update_active_set=frequency_update_active_set,
		              callbackFunc=callbackFunc,max_time=max_time)
			
		elif method=="DualGradientAscent":
			x, y_eq,y_ineq=DualGradientAscent(x=x0,LP=self,nbmaxiter=nb_iter,callbackFunc=callbackFunc,y_eq=None,y_ineq=None,max_time=max_time)
		elif method=="DualCoordinateAscent":
			x, y_eq,y_ineq=DualCoordinateAscent(x=x0,LP=self,nbmaxiter=nb_iter,callbackFunc=callbackFunc,y_eq=None,y_ineq=None,max_time=max_time)

		else: 
			print "unkown LP solver method "+method
			raise
		elapsed= (time.clock() - start)		
		
		if getTiming:
			
			return x,elapsed
		else:
			return x
		

