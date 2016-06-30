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
import scipy.sparse
import time

class chrono():
	def __init__(self):
		pass
	def tic(self):
		self.start=time.clock() 
	def toc(self):
		return time.clock()-self.start 
	
	
class check_decrease:
	def __init__(self,val=None,tol=1e-10):
		self.val=val
		self.tol=tol
	def set(self,val):
		self.val=val
	def add(self,val):
		assert(self.val>=val-self.tol)
		self.val=val
		
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
	
	if not Aineq is None:
		ni=Aineq.shape[0]
		#need to convert in standard form by adding an auxiliary variables for each inequality
		if not Aeq is None:
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
	if (not Aineq is None) and (not b_lower is None):
	
	
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



class solutionStat():
	
	def __init__(self,c,AeqCSC,beq,AineqCSC,bineq,callbackFunc):
		self.c=c
		self.Aeq=AeqCSC
		self.beq=beq
		self.Aineq=Aineq
		self.bineq=bineq
		self.best_integer_solution_energy=np.inf
		self.best_integer_solution=None
		self.iprev=0
		self.callbackFunc=callbackFunc
		
	def startTimer(self):
		self.start = time.clock() 				
		self.elapsed=	self.start
		
	
		
	def eval(self,x,i):
		
		self.prev_elapsed=self.elapsed
		self.elapsed= (time.clock() - start)	
		nb_iter_since_last_call=i-self.self.iprev
		mean_iter_period=(elapsed-prev_elapsed)/nb_iter_since_last_call
		
		
		energy1=self.c.dot(x)
		max_violated_equality=0
		max_violated_inequality=0
		r_eq=(self.Aeq*x)-self.beq
		r_ineq=(self.Aineq*x)-self.bineq			
		if not self.Aeq is None:							
			max_violated_equality=np.max(np.abs(r_eq))
		if not self.Aineq is None: 
			max_violated_inequality=np.max(r_ineq)
	
		xrounded=np.round(x)
		energy_rounded=c.dot(xrounded)
		nb_violated_equality_rounded=np.sum(np.abs(Aeq*xrounded-beq))
		nb_violated_inequality_rounded=np.sum(np.maximum(Aineq*xrounded-bineq,0))	
	
		if nb_violated_equality_rounded==0 and nb_violated_inequality_rounded==0:
			print '##########   found feasible solution with energy'+str(energy_rounded)
			if energy_rounded<best_integer_solution_energy:
				self.best_integer_solution_energy=energy_rounded
				self.best_integer_solution=xrounded


		print 'iter'+str(i)+": energy1= "+str(energy1) +  ' elaspsed '+str(elapsed)+' second'+\
	              ' max violated inequality:'+str(max_violated_inequality)+\
	              ' max violated equality:'+str(max_violated_equality)+\
	              'mean_iter_period='+str(mean_iter_priod)+\
	              'rounded : %f ineq %f eq'%(nb_violated_inequality_rounded,nb_violated_equality_rounded)
			#'y_eq has '+str(100 * np.mean(y_eq==0))+' % of zeros '+\
		#    'y_ineq has '+str(100 * np.mean(y_ineq==0))+' % of zeros '+\
		self.iprev=i
	
		if self.callbackFunc!=None:
	
			self.callbackFunc(i,x,energy1,energy2,elapsed,max_violated_equality,max_violated_inequality,is_active_variable=is_active_variable)
	
		

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





def preconditionConstraints(A,b,b2=None,alpha=2):
	#alpha=2
	ACopy=A.copy()
	ACopy.data=np.abs(ACopy.data)**(alpha)
	SumA=ACopy*np.ones((ACopy.shape[1]))
	tmp=(SumA)**(1./alpha)
	tmp[tmp==0]=1
	diagSigmA=1/tmp
	Sigma=scipy.sparse.diags([diagSigmA],[0]).tocsr()
	Ap=Sigma*A
	Ap.__dict__['blocks']=A.blocks
	bp=Sigma*b
	if b2==None:

		return Ap,bp
	else:
		return Ap,bp,Sigma*b2



def preconditionLPRight(c,Aeq,beq,lb,ub,x0,alpha=2):
	#alpha=2
	AeqCopy=Aeq.copy()
	AeqCopy.data=np.abs(AeqCopy.data)**(alpha)
	SumA=np.ones((AeqCopy.shape[0]))*AeqCopy
	tmp=(SumA)**(1./alpha)
	tmp[tmp==0]=1
	diagR=1/tmp
	R=scipy.sparse.diags([diagR],[0]).tocsr()

	Aeq2=Aeq*R
	beq2=beq
	lb2=tmp*lb
	ub2=tmp*ub
	x02=tmp*x0
	c2=c*R
	Aeq2.__dict__['blocks']=Aeq.blocks

	return R,c2,Aeq2,beq2,lb2,ub2,x02
