class chrono():
	def __init__(self):
		pass
	def tic(self):
		self.start=time.clock() 
	def toc(self):
		return time.clock()-self.start 
def convertToPySparseFormat(A):
	#check symetric
	assert((A-A.T).nnz==0)
	L = spmatrix.ll_mat_sym(A.shape[0],A.nnz )
	Acoo=scipy.sparse.triu(A).tocoo()
	L.put(Acoo.data, Acoo.row.astype(int), Acoo.col.astype(int))

	return L

class CholeskyOrLu():
	def __init__(self,M,type):
		if type=='scipySparseLu':
		
			self.LU=scipy.sparse.linalg.splu(M.tocsc())
			self.solve=self.LU.solve
		elif type=='scikitsCholesky':
			self.LU=   scikits.sparse.cholmod.cholesky(M.tocsc())
			self.solve =self.LU.solve_A	
		elif type=='umfpackLU':
			M2=convertToPySparseFormat(M)
			self.LU_umfpack = umfpack.factorize(M2, strategy="UMFPACK_STRATEGY_SYMMETRIC")
			print "nnz per line :"+str(LU_umfpack.nnz/float(M2.shape[0]) ) 
			print "factorization :"+str(c.toc())
			
			LU_umfpack.solve(b,x)		

def convertToStandardFormWithBounds(c,Aeq,beq,Aineq,b_lower,b_upper,lb,ub,x0):
	
	if Aineq!=None:
		ni=Aineq.shape[0]
		#need to convert in standard form by adding an auxiliary variables for each inequality
		if Aeq!=None:
			Aeq2=scipy.sparse.vstack((scipy.sparse.hstack((Aeq,scipy.sparse.csc_matrix((Aeq.shape[0], ni)))),scipy.sparse.hstack((Aineq,-scipy.sparse.eye(ni,ni))))).tocsr()
			Aeq2.__dict__['blocks']=Aeq.blocks+[(b[0]+Aeq.shape[0],b[1]+Aeq.shape[0]) for b in Aineq.blocks]
			beq2=np.hstack((beq,np.zeros((ni))))
		else:
			
			Aeq2=scipy.sparse.hstack((Aineq,-scipy.sparse.eye(ni,ni))).tocsr()
			Aeq2.__dict__['blocks']=Aineq.blocks
			beq2=np.zeros((ni))

		lb=np.hstack((lb,b_lower))
		ub=np.hstack((ub,b_upper))		
		epsilon0=Aineq*x0
		x0=np.hstack((x0,epsilon0))
		c=np.hstack((c,np.zeros(ni)))	
	return c,Aeq2,beq2,lb,ub,x0

def convertToOnesideInequalitySystem(Aineq,b_lower,b_upper):
	if Aineq!=None and b_lower!=None:
	
	
		idskeep_upper=np.nonzero(b_upper!=np.inf)[0]
		idskeep_lower=np.nonzero(b_lower!=-np.inf)[0]
		if len(idskeep_lower)>0 and len(idskeep_upper)>0:
			Aineq=scipy.sparse.vstack((Aineq[idskeep_upper,:],-Aineq[idskeep_lower,:])).tocsr()
		elif len(idskeep_lower)>0 :
			Aineq=-Aineq
		else:
			Aineq=Aineq
		bineq=np.hstack((b_upper[idskeep_upper],-b_lower[idskeep_lower]))
	else:
		bineq=b_upper
	return Aineq,bineq

def check_constraints(i,x_r,mask,Acsr,Acsc,b_lower,b_upper):
	violated=False
	constraints_to_check=np.nonzero(Acsc[:,i])[0]
	for j in constraints_to_check:
		line=Acsr[j,:]
		interval_d=0
		interval_u=0
		for k in range(line.indices.size):
			i=line.indices[k]
			v=line.data[k]
	
			if mask[i]>0:
				interval_d+=v*x_r[i]
				interval_u+=v*x_r[i]
			elif v>0:
				interval_u+=v
			else: 
				interval_d+=v
		if interval_u<b_lower[j] or interval_d>b_upper[j]:
			violated=True
			break
	return violated

def save_arguments(filename):
	"""Returns tuple containing dictionary of calling function's
	   named arguments and a list of calling function's unnamed
	   positional arguments.
	"""
	from inspect import getargvalues, stack
	import inspect
	posname, kwname, args = getargvalues(stack()[1][0])[-3:]
	posargs = args.pop(posname, [])
	args.update(args.pop(kwname, []))
	caller = inspect.currentframe().f_back
	func_name=(caller.f_code.co_name)
	
	module= caller.f_globals['__name__']	
	import pickle
	d={'module':module,'function_name':func_name,'args':args,'posargs':posargs}
	with open(filename, 'wb') as f:				
		pickle.dump(d,f)
