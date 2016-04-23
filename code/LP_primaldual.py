import debug_tools
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

import scipy.sparse
import scipy.sparse.linalg
from cython_tools import *
import pyamg 
import scikits.sparse.cholmod
from pysparse import spmatrix
from pysparse.direct import umfpack
import cython_tools
from scipy.misc import imsave
#from projsplx import projsplx



	




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
	
#@profile		
def LP_admm(c,Aeq,beq,Aineq,b_lower,b_upper,lb,ub,x0=None,gamma_eq=2,gamma_ineq=3,nb_iter=100,callbackFunc=None,max_time=None,use_preconditionning=True):
	# simple ADMM method with an approximate resolutio of a quadratic subproblem using conjugate gradient
	useLU=False
	useCholesky=False
	useAMG=False
	useCG=False
	useBoundedGaussSiedel=True
	useUnboundedGaussSiedel=False	
	
	
	n=c.size
	if x0==None:
		x0=np.zeros(c.size)
	if Aeq!=None:
		Aeq,beq=preconditionConstraints(Aeq,beq,alpha=2)
	if Aineq!=None:# it seem important to do this preconditionning before converting to standard form
			Aineq,b_lower,b_upper=preconditionConstraints(Aineq,b_lower,b_upper,alpha=2)
	c,Aeq,beq,lb,ub,x0=	convertToStandardFormWithBounds(c,Aeq,beq,Aineq,b_lower,b_upper,lb,ub,x0)
	x=x0

	#trying some preconditioning
	if use_preconditionning:
		Aeq,beq=preconditionConstraints(Aeq,beq,alpha=2)



	AtA=Aeq.T*Aeq
	#AAt=Aeq*Aeq.T
	Atb=Aeq.T*beq
	Id=scipy.sparse.eye(x.size,x.size)

	xp=np.maximum(x,0)

	M=gamma_eq*AtA+gamma_ineq*Id
	M=M.tocsr()
	lambda_eq=np.zeros(Aeq.shape[0])
	lambda_ineq=np.zeros(x.shape)
	if useLU:
		luM = scipy.sparse.linalg.splu(M)
		#luM = scipy.sparse.linalg.spilu(M,drop_tol=0.01)
	elif useCholesky:
		ch=chrono()
		ch.tic()
		Chol=     scikits.sparse.cholmod.cholesky(M.tocsc())
		print 'cholesky factorization took '+str(ch.toc())+ ' seconds'
		print 'the sparsity ratio between the cholesky decomposition of M and M is '+str(Chol.L().nnz/float(M.nnz))
		
	elif useAMG:
		Mamg=pyamg.ruge_stuben_solver(M)
	def L(x,xp,lambda_eq,lambda_ineq):
		E=c.dot(x)\
	                +0.5*gamma_eq*np.sum((Aeq*x-beq)**2)\
	                +0.5*gamma_ineq*np.sum((x-xp)**2)\
	                +lambda_eq.dot(Aeq*x-beq)\
	                +lambda_ineq.dot(x-xp)
		return E
	i=0
	nb_cg_iter=1
	speed=np.zeros(x.shape)

	order=np.arange(x.size).astype(np.uint32)
	bs=boundedGaussSeidelClass(M)
	alpha=1.4
	start = time.clock() 				
	elapsed=start	
	while i <=nb_iter/nb_cg_iter:
		#solve the penalized problem with respect to x
		#c +gamma_eq*(AtA x-Atb) + gamma_ineq*(x -xp)+lambda_eq*Aeq+lambda_ineq
		# M*x=-c+Atb+gamma_ineq*xp-lambdas-lambda_eq*Aeq
		
		y=-c+gamma_eq*Atb+gamma_ineq*xp- lambda_eq*Aeq-lambda_ineq
		#print 'iter'+str(i)+' '+str(L(x, xp,lambda_eq,lambda_ineq))	
		if useLU:
			x=luM.solve(y)
		elif useCholesky:
			x=Chol.solve_A(y)
		elif useBoundedGaussSiedel:
			xprev=x.copy()	
			
			#x=xprev+1*speed	# maybe could do a line search along that direction ? 	
			#if i%2==0:
				#order=np.arange(x.size).astype(np.uint32)
			#else:
				#order=np.arange(x.size-1,-1,-1).astype(np.uint32)
			bs.solve(y,lb,ub, x,maxiter=nb_cg_iter,w=1,order=order)	
			speed=x-xprev
		elif useUnboundedGaussSiedel:
			xprev=x.copy()	
			#x=xprev+1*speed	# predict the minimum , can yield to some speedup	
			if False:# optimal sep along the direction given by the last two iterates, does not seem to imrove much speed
					direction=speed
					t=-direction.dot(M*xprev-y)
					print t
					if abs(t)>0:
						step_lenght=t/(direction.dot(M*direction))
						x=xprev+step_lenght*direction		
			else:
				pass
				#x=xprev+0.8*speed
			GaussSeidel(M,y, x,maxiter=nb_cg_iter,w=1.0)
			speed=x-xprev
			x=alpha*x+(1-alpha)*xp
		elif useCG:
			#x1,r=scipy.sparse.linalg.cgs(M, y,  maxiter=2,x0=x)
			xprev=x.copy()	
			
			if True:# optimal sep along the direction given by the last two iterates, doe not seem to improve things in term of numbe rof iteration , and slow down iterations...
				# maybe use next step as a conguate step would help ? 
				direction=speed
				t=-direction.dot(M*x-y)
				if abs(t)>0:
					step_lenght=t/(direction.dot(M*direction))
					x=x+step_lenght*direction			
			else:
				#x=xprev+1*speed			# does not work with cg, explode
				pass
			#start conjugate gradient from there (could use previous direction ? )
			x=conjgrad(M, y,  maxiter=nb_cg_iter,x0=x)
			speed=x-xprev
			x=alpha*x+(1-alpha)*xp
		elif useAMG:
			#xprev=x.copy()	
			#x=xprev+1*speed	
			x = Mamg.solve(y, x0=x,tol=1e-3) 
			#speed=x-xprev
			x=alpha*x+(1-alpha)*xp#over relaxation
			
		else:
			print 'unkown method'
			raise
		
		if i%10==0:
			prev_elapsed=elapsed
			elapsed= (time.clock() - start)	
			energy1=L(x, xp,lambda_eq,lambda_ineq)
			energy2=energy1
			r=Aeq*x-beq
			max_violated_equality=np.max(np.abs(r))
			max_violated_inequality=max(0,-np.min(x))
			print 'iter'+str(i)+": energy1= "+str(energy1) + " energy2="+str(energy2)+ ' elaspsed '+str(elapsed)+' second'+\
			      ' max violated inequality:'+str(max_violated_inequality)+\
			      ' max violated equality:'+str(max_violated_equality)
			if callbackFunc!=None:
				callbackFunc(i, x[0:n],energy1,energy2,elapsed,max_violated_equality,max_violated_inequality)			
			
		#solve the penalized problem with respect to xp	
		#-gamma_ineq*(x-xp)-lambda_ineq=0
		if not(useBoundedGaussSiedel):
			xp=x.copy()+lambda_ineq/gamma_ineq
			xp=np.maximum(xp,lb)
			xp=np.minimum(xp,ub)
			lambda_ineq=lambda_ineq+gamma_ineq*(x-xp)
			#print np.max(np.abs(lambda_ineq))
		else:
			xp=x
		#print 'iter'+str(i)+' '+str(L(x, xp,lambda_eq,lambda_ineq))
		lambda_eq=lambda_eq+gamma_eq*(Aeq*x-beq)# could use heavy ball instead of gradient step ? 
		
		#could try to update the penality ? 
		#gamma_ineq=gamma_ineq+
		#M=gamma_eq*AtA+gamma_ineq*Id
		i+=1
	return x[0:n]




def LP_admm2(c,Aeq,beq,Aineq,b_lower,b_upper,lb,ub,x0=None,gamma_ineq=0.7,nb_iter=100,callbackFunc=None,max_time=None,use_preconditionning=False):
	# simple ADMM method with an approximate resolution of a quadratic subproblem using conjugate gradient
	# inspiredy by Boyd's paper on ADMM
	# Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
	# the difference with LP_admm is that the linear quality constrainrs Aeq*beq are enforced during the resolution 
	# of the subproblem instead of beeing enforced through multipliers 
	useLU=False
	useAMG=False
	useCholesky=True
	useCholesky2=False
	
	alpha=1.95#relaxation paramter should be in [0,2] , 1.95 seems to be often a good choice
	
	start = time.clock() 	
	elapsed=start	
	n=c.size
	if x0==None:
		x0=np.zeros(c.size)
	
	if use_preconditionning:
		if Aeq!=None:
			Aeq,beq=preconditionConstraints(Aeq,beq,alpha=2)
		if Aineq!=None:# it seem important to do this preconditionning before converting to standard form
			Aineq,b_lower,b_upper=preconditionConstraints(Aineq,b_lower,b_upper,alpha=2)
			
	c,Aeq,beq,lb,ub,x0=	convertToStandardFormWithBounds(c,Aeq,beq,Aineq,b_lower,b_upper,lb,ub,x0)
	x=x0
	
	xp=x.copy()
	xp=np.maximum(xp,lb)
	xp=np.minimum(xp,ub)	
	ch=chrono()
	#trying some preconditioning
	if use_preconditionning:
		Aeq,beq=preconditionConstraints(Aeq,beq,alpha=2)


	M=scipy.sparse.vstack((scipy.sparse.hstack((gamma_ineq*scipy.sparse.eye(Aeq.shape[1],Aeq.shape[1]),Aeq.T)),\
                               scipy.sparse.hstack((Aeq, scipy.sparse.csc_matrix((Aeq.shape[0], Aeq.shape[0])))))).tocsr()
	if useLU:
		luM = scipy.sparse.linalg.splu(M.tocsc()) 
		nb_cg_iter=1
	elif useCholesky:
		
		ch.tic()
		Chol=     scikits.sparse.cholmod.cholesky(M.tocsc(),mode='simplicial')
		print 'cholesky factorization took '+str(ch.toc())+ ' seconds'
		print 'the sparsity ratio between the cholesky decomposition of M and M is '+str(Chol.L().nnz/float(M.nnz))
		nb_cg_iter=1
	elif useCholesky2:	
		print "using UMFPACK_STRATEGY_SYMMETRIC through PySparse"
		ch.tic()
		M2=convertToPySparseFormat(M)
		print "conversion :"+str(ch.toc())
		ch.tic()
		LU_umfpack = umfpack.factorize(M2, strategy="UMFPACK_STRATEGY_SYMMETRIC")
		print "nnz per line :"+str(LU_umfpack.nnz/float(M2.shape[0]) ) 
		print "factorization :"+str(ch.toc())
		nb_cg_iter=1
		
	elif useAMG:
		#Mamg=pyamg.smoothed_aggregation_solver(M.tocsc())
		#Mamg=pyamg.rootnode_solver(M.tocsc())
		#Mamg=pyamg.
		Mamg=pyamg.ruge_stuben_solver(M.tocsc(),strength=None)#sometimes seems to yield infinte values
		#I=scipy.sparse.eye(1000)
		#Mamg=pyamg.ruge_stuben_solver(I.tocsc())
		
		for l in range(len(Mamg.levels)):
			print "checking level " + str(l)
			assert np.isfinite(Mamg.levels[l].A.data).all()
		nb_cg_iter=1
	else:
		nb_cg_iter=100
	lambda_ineq=np.zeros(x.shape)
                            
	def L(x,xp,lambda_ineq):
		E=c.dot(x)\
	                +0.5*gamma_ineq*np.sum((x-xp)**2)\
	                +lambda_ineq.dot(x-xp)
		return E
	i=0
	xv=np.hstack((x,np.zeros(beq.shape)))

	while i <=nb_iter/nb_cg_iter:
		#solve the penalized problem with respect to x		
		#print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))	
		
		y=np.hstack((-c+gamma_ineq*xp- lambda_ineq,beq))
		if useLU:
			xv=luM.solve(y)
		elif useCholesky:
			xv=Chol.solve_A(y)
		elif useCholesky2:
			
		
			LU_umfpack.solve(y,xv)
					
		elif useAMG:
			xv = Mamg.solve(y, x0=xv,tol=1e-12)
			if np.linalg.norm(M*xv-y)>1e-5:
				raise
			
		else:
			xv=conjgrad(M, y,  maxiter=nb_cg_iter,x0=xv)
		x=xv[:x.shape[0]]
		x=alpha*x+(1-alpha)*xp
		
		#print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))		
		#solve the penalized problem with respect to xp	
		#-gamma_ineq*(x-xp)-lambda_ineq=0
		xp=x.copy()+lambda_ineq/gamma_ineq
		xp=np.maximum(xp,lb)
		xp=np.minimum(xp,ub)
		if i%10==0:
			prev_elapsed=elapsed
			elapsed= (time.clock() - start)	
			energy1=L(x, xp,lambda_ineq)
			energy2=energy1
			
			max_violated_equality=0
			max_violated_inequality=0
			print 'iter'+str(i)+": energy1= "+str(energy1) + " energy2="+str(energy2)+ ' elaspsed '+str(elapsed)+' second'+\
			      ' max violated inequality:'+str(max_violated_inequality)+\
			      ' max violated equality:'+str(max_violated_equality)
			if callbackFunc!=None:
				callbackFunc(i, x[0:n],energy1,energy2,elapsed,max_violated_equality,max_violated_inequality)		
		
		#print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))
		lambda_ineq=lambda_ineq+gamma_ineq*(x-xp)
		i+=1
	return x[0:n]

class check_decrease:
	def __init__(self,val=None,tol=1e-10):
		self.val=val
		self.tol=tol
	def set(self,val):
		self.val=val
	def add(self,val):
		assert(self.val>=val-self.tol)
		self.val=val
		
	
def LP_admmBlockDecomposition(c,Aeq,beq,Aineq,b_lower,b_upper,lb,ub,x0=None,gamma_ineq=0.7,nb_iter=100,callbackFunc=None,max_time=None,use_preconditionning=True,useLU=True,nb_iter_plot=10):
	# simple ADMM method with an approximate resolutio of a quadratic subproblem using conjugate gradient
	# inspiredy by Boyd's paper on ADMM
	# Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
	# the difference with LP_admm is that the linear quality constrainrs Aeq*beq are enforced during the resolution 
	# of the subproblem instead of beeing enforce through multipliers 
	n=c.size
	start = time.clock() 				
	elapsed=start	
	if x0==None:
		x0=np.zeros(c.size)
	#if Aeq!=None:
		#Aeq,beq=preconditionConstraints(Aeq,beq,alpha=2)
	#if Aineq!=None:# it seem important to do this preconditionning before converting to standard form
		#Aineq,b_lower,b_upper=preconditionConstraints(Aineq,b_lower,b_upper,alpha=2)

	c,Aeq,beq,lb,ub,x0=	convertToStandardFormWithBounds(c,Aeq,beq,Aineq,b_lower,b_upper,lb,ub,x0)
	x=x0

	xp=x.copy()
	xp=np.maximum(xp,lb)
	xp=np.minimum(xp,ub)	

	#trying some preconditioning
	use_preconditionning_rows=False # left preconditioning seems not to change anything if not used in combination with use_preconditionning_cols  as each subproblem is solved exactly.
	if use_preconditionning_rows:
		Aeq,beq=preconditionConstraints(Aeq,beq,alpha=2)
		
	#global right preconditioning
	use_preconditionning_cols=False 
	if use_preconditionning_cols:
		R,c,Aeq,beq,lb,ub,x0=preconditionLPRight(c,Aeq,beq,lb,ub,x0,alpha=3)
	else:
		R=sparse.eye(Aeq.shape[1])

	luMs=[]
	nb_used=np.zeros(x.shape)
	list_block_ids=[]
	#Aeq.blocks=[(0,Aeq.blocks[5][1]),(Aeq.blocks[6][0],Aeq.blocks[11][1])]	
	#Aeq.blocks=[(0,Aeq.blocks[-1][1])]	
	
	beqs=[]
	usesparseLU=False
	xv=[]
	#for idblock in range(nb_blocks):
	ch=chrono()	
	
	mergegroupes=[]
	mergegroupes=[[k] for k in range(len(Aeq.blocks))]
	#mergegroupes.append(np.arange(len(Aeq.blocks)))#single block
	
	#mergegroupes.append([0,1,2,3, 4, 5])
	#mergegroupes.append([6,7,8,9,10,11])
	#mergegroupes.append([12,13,14])
	#mergegroupes.append([0,1,2,3, 4, 5,12])
	#mergegroupes.append([6,7,8,9,10,11,13])
	#mergegroupes.append([14])	
	
	#mergegroupes.append([0,1,2,3,4,5,12,13,])
	#mergegroupes.append([6,7,8,9,10,11,14,15])
	#mergegroupes.append([0,1,2,3,4,5])
	#mergegroupes.append([6,7,8,9,10,11])
	#mergegroupes.append([12,13])
	#mergegroupes.append([14,15])
	#mergegroupes.append([12,13,14,15])
	nb_blocks=len( mergegroupes)
	
	if False:
		idRows=np.hstack([np.arange(Aeq.blocks[g][0],Aeq.blocks[g][1]+1) for g in mergegroupe])
		
		# we want to cluster the constraints such that the submatrix for each cluster has a sparse cholesky decomposition
		# we need to reach a tradeoff between the number of cluster (less variables copies) and the sparsity of the cholesky decompositions
		# each cluster should have a small tree width ? 
		# can we do an incremental sparse cholseky an then  add one constraint to each cholesky at a time ? 
		# is non sparse cholseky decomposition  (without permutation)incremental ? 
		# adding a factor to a block is good if does not add too many new variables to the block and does not augment the treewidth of the block ? 
		# we can do incremental merging of blocks
		# marge is good if the interection of the variables used by the two block is large (remove copies) (fast to test)
		# and if the operation does not increase the cholseky density or graph tree width too much 
		
		subA=Aeq[idRows,:]
		t=np.array(np.abs(subA).sum(axis=0)).ravel()
		ids=np.nonzero(t)[0]
		list_block_ids.append(ids)
	
		subA2=subA[:,ids]
		# precompute the LU factorizartion of the matrix that needs to be inverted for the block
		M=scipy.sparse.vstack((scipy.sparse.hstack((gamma_ineq*scipy.sparse.eye(subA2.shape[1],subA2.shape[1]),subA2.T)),\
			                       scipy.sparse.hstack((subA2, scipy.sparse.csc_matrix((subA2.shape[0], subA2.shape[0])))))).tocsr()
		LU=     scikits.sparse.cholmod.cholesky(M.tocsc(),mode='simplicial')
		print 'the sparsity ratio between Chol(M) and the  matrix M  is +'+str(LU.L().nnz/float(M.nnz))
		
		
		print 'connected components M'+str(scipy.sparse.csgraph.connected_components(M))
			
		#scipy.sparse.csgraph.connected_components(subA.T*subA)
		FactorConnections=subA*subA.T
		print 'connected components F'+str(scipy.sparse.csgraph.connected_components(FactorConnections))
		ST=scipy.sparse.csgraph.minimum_spanning_tree(FactorConnections)
		
	
	
	for idblock,mergegroupe in enumerate(mergegroupes):
		# find the indices of the variables used by the block
		idRows=np.hstack([np.arange(Aeq.blocks[g][0],Aeq.blocks[g][1]+1) for g in mergegroupe])
		subA=Aeq[idRows,:]
		t=np.array(np.abs(subA).sum(axis=0)).ravel()
		ids=np.nonzero(t)[0]
		list_block_ids.append(ids)
		# increment th number of time each variable is copied
		nb_used[ids]+=1
		subA2=subA[:,ids]
		# precompute the LU factorizartion of the matrix that needs to be inverted for the block
		M=scipy.sparse.vstack((scipy.sparse.hstack((gamma_ineq*scipy.sparse.eye(subA2.shape[1],subA2.shape[1]),subA2.T)),\
				       scipy.sparse.hstack((subA2, scipy.sparse.csc_matrix((subA2.shape[0], subA2.shape[0])))))).tocsr()
		if usesparseLU:
			
			ch.tic()
			LU=scipy.sparse.linalg.splu(M.tocsc())
			print ch.toc()
		else:
			ch.tic()
			LU=     scikits.sparse.cholmod.cholesky(M.tocsc(),mode='simplicial')
			factorization_duration= ch.toc()
			#A=     scikits.sparse.cholmod.analyze(M.tocsc(),mode='simplicial')
			#P=scipy.sparse.coo_matrix((np.ones(A.P().size),(A.P(),np.arange(A.P().size))))
			#permuted=P*M*P.T
			#plt.imshow(permuted.todense())
		# A.P()
			
			#LU=     scikits.sparse.cholmod.cholesky(M.tocsc(),mode='supernodal')# gives me matrix is not positive definite errors..
			
			print 'the sparsity ratio between Chol(M) and the  matrix M for block'+str(idblock)+' is +'+str(LU.L().nnz/float(M.nnz))+\
			' took '+str(factorization_duration)+ 'seconds to factorize'
			
			
			#LU.__=LU.solve_A
			#M2=convertToPySparseFormat(M)
			#LU = umfpack.factorize(M2, strategy="UMFPACK_STRATEGY_SYMMETRIC")
			#print "nnz per line :"+str(LU.nnz/float(M2.shape[0]) ) 
			
			
		xv.append(np.empty(M.shape[1],dtype=float))					
		luMs.append(LU )
		beqs.append(beq[idRows])
		pass
		
	
	

	def L(x,xp,lambda_ineq):
		E=c.dot(xp)
		for idblock in range(nb_blocks):
			diff=x[idblock]-xp[list_block_ids[idblock]]
			E+=0.5*gamma_ineq*np.sum((diff)**2)\
			+lambda_ineq[idblock].dot(diff)
		return E
	
	i=0
	
	x=[x0[list_block_ids[i]] for i in range(nb_blocks)]
	lambda_ineq=[np.zeros(list_block_ids[i].shape) for i in range(nb_blocks)]
	

	check=check_decrease(tol=1e-10)
	dpred=[np.zeros(list_block_ids[i].shape) for i in range(nb_blocks)]
	alpha=1.95#relaxation paramter should be in [0,2] , 1.95 seems to be often a good choice

	while i <=nb_iter:
		#solve the penalized problems with respect to each copy x		
		#print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))
		#check.set(L(x, xp,lambda_ineq))
		for idblock in range(nb_blocks):
			y=np.hstack((gamma_ineq*xp[list_block_ids[idblock]]- lambda_ineq[idblock],beqs[idblock]))
			if usesparseLU:
				xv[idblock]=luMs[idblock].solve(y)	
			else:
				xv[idblock]=luMs[idblock].solve_A(y)	
				
				#luMs[idblock].solve(y,xv[idblock])
			x[idblock]=alpha*xv[idblock][:x[idblock].shape[0]]+(1-alpha)*xp[list_block_ids[idblock]]
			
			#check.add(L(x, xp,lambda_ineq))
		#print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))		
		#solve the penalized problem with respect to xp	
		#c-sum_idblock  gamma_ineq*(x_[idblock]-xp[list_block_ids[idblock]])-lambda_ineq[idblock]=0
		xp[nb_used>0]=0
		for idblock in range(nb_blocks):
			xp[list_block_ids[idblock]]+=x[idblock]+lambda_ineq[idblock]/gamma_ineq # change formula here
		
		xp=xp-c/gamma_ineq 
		xp=xp/	np.maximum(nb_used,1)
		xp=np.maximum(xp,lb)
		xp=np.minimum(xp,ub)
		#check.add(L(x, xp,lambda_ineq))
		
		for idblock in range(nb_blocks):
			d=gamma_ineq*(x[idblock]-xp[list_block_ids[idblock]])
			#angle=np.sum(dpred[idblock]*d)/(np.sqrt(np.sum(dpred[idblock]**2))*+np.sqrt(np.sum(d**2)))
			#print angle
			#dpred[idblock]=d.copy()
			lambda_ineq[idblock]=lambda_ineq[idblock]+d
			#lambda_ineq[idblock]=lambda_ineq[idblock]+(1+max(0,angle))*d # trying some naive heuristic speedup but not working :(
			
			
		if i%nb_iter_plot==0:
					prev_elapsed=elapsed
					elapsed= (time.clock() - start)	
					if elapsed>max_time:
						break
					
					energy1=L(x, xp,lambda_ineq)
					energy2=energy1
					
					max_violated_equality=0
					max_violated_inequality=0
					print 'iter'+str(i)+": energy1= "+str(energy1) + " energy2="+str(energy2)+ ' elaspsed '+str(elapsed)+' second'+\
				          ' max violated inequality:'+str(max_violated_inequality)+\
				          ' max violated equality:'+str(max_violated_equality)
					if callbackFunc!=None:
						callbackFunc(i,(R* xp)[0:n],energy1,energy2,elapsed,max_violated_equality,max_violated_inequality)				
		i+=1
				
	return (R* xp)[0:n]


				
			
	
	


	


def revert(back_ops,x_l,x_u):
	for t,i,v in reversed(back_ops):
		if t==0:
			x_l[i]=v
		else:
			x_u[i]=v


def dualCoordinateAscent(x,LP,nbmaxiter=20,callbackFunc=None,y_eq=None,y_ineq=None):
	"""Method from 'An algorigthm for large scale 0-1 integer 
	programming with application to ailine crew scheduling'
	we generelized it to the case where A is not 0-1 and
	the upper bounds can be greater than 1
	did not generalize and code the approximation method
	"""
	# convert to slack form (augmented form)
	LP2=copy.deepcopy(LP)
	LP=None
	#LP2.convertToSlackForm()
	#LP2.convertTo
	LP2.convertToOnesideInequalitySystem()
	
	#LP2.upperbounds=np.minimum(10,LP2.upperbounds)
	#LP2.lowerbounds=np.maximum(-10,LP2.lowerbounds)
	#y0=None
	if y_eq is None:
		y_eq=np.zeros(LP2.Aequalities.shape[0])
		#y_eq=-np.random.rand(y_eq.size)
	else:
		y_eq=y_eq.copy()
	#y_ineq=None
	if y_ineq is None:
		y_ineq=np.zeros(LP2.Ainequalities.shape[0])
		y_ineq=np.abs(np.random.rand(y_ineq.size))
	else:
		y_ineq=y_ineq.copy()
		assert(np.min(y_ineq)>=0)
	#assert (LP2.B_lower is None)
	
	def getOptimX(y_eq,y_ineq):
		c_bar=LP2.costsvector.copy()
		if not LP2.Aequalities is None:
			c_bar+=y_eq*LP2.Aequalities
		if not LP2.Ainequalities is None:
			c_bar+=y_ineq*LP2.Ainequalities
		x=np.zeros(LP2.costsvector.size)
		x[c_bar>0]=LP2.lowerbounds[c_bar>0]
		x[c_bar<0]=LP2.upperbounds[c_bar<0]
		x[c_bar==0]=0.5*(LP2.lowerbounds+LP2.upperbounds)[c_bar==0]
		return c_bar,x	
	
	def eval(y_eq,y_ineq):
		c_bar,x	=getOptimX(y_eq,y_ineq)
		E=-y_eq.dot(LP2.Bequalities)-y_ineq.dot(LP2.B_upper)+np.sum(x*c_bar)
		#LP2.costsvector.dot(x)+y_ineq.dot(LP2.Ainequalities*x-LP2.B_upper)
		E=-y_eq.dot(LP2.Bequalities)-y_ineq.dot(LP2.B_upper)+np.sum(np.minimum(c_bar*LP2.upperbounds,c_bar*LP2.lowerbounds)[c_bar!=0])
		return E
	
	def exactCoordinateLineSearch(Ai,bi,c_bar):
		alphas=-c_bar[Ai.indices]/Ai.data
		order=np.argsort(alphas)
		AiU=Ai.data*LP2.upperbounds[Ai.indices]
		AiL=Ai.data*LP2.lowerbounds[Ai.indices]
		tmp1=np.minimum(AiU[order], AiL[order])
		tmp2=np.maximum(AiU[order], AiL[order])
		tmp3=np.cumsum(tmp2[::-1])[::-1]
		tmp4=np.cumsum(tmp1)
		derivatives=-bi*np.ones(alphas.size+1)
		derivatives[:-1]+=tmp3
		derivatives[1:]+=tmp4
	
		#tmp=np.abs(Ai.data[order])*(LP2.lowerbounds[Ai.indices[order]]-LP2.upperbounds[Ai.indices[order]])
		#derivatives= -LP2.Bequalities[i]-np.sum(AiL[Ai.data>0])-np.sum(AiU[Ai.data<0])\
			#+np.hstack(([0],np.cumsum(tmp)))
	
		k=np.searchsorted(-derivatives,0)
		if derivatives[k]==0:
			t=np.random.rand()
			alpha_optim=t*alphas[order[k]]+(1-t)*alphas[order[k-1]]#maybe courld draw and random valu in the interval ? 
		else:
			alpha_optim=alphas[order[k-1]]	
		return alpha_optim
	#x[c_bar==0]=0.5
	
	# alpha_i= vector containing the step lenghts that lead to a sign change on any of the gradient component 
	# when incrementing y[i]
	#
	print 'iter %d energy %f'%(0 ,eval(y_eq,y_ineq))
	c_bar,x=getOptimX(y_eq, y_ineq)
	for iter in range(nbmaxiter):
		y_ineq_prev=y_ineq.copy()
		for i in range(y_eq.size):
		
			#i=32
			#print eval(y)
			if False:
				y2=y_eq.copy()
				vals=[]
				alphasgrid=np.linspace(-5,5,1000)
				for alpha in alphasgrid:
					y2[i]=y_eq[i]+alpha	
					vals.append(eval(y2,y_ineq))
				plt.plot(alphasgrid,vals,'.')
				deriv=np.diff(vals)/np.diff(alphasgrid)
				plt.plot(alphasgrid[:-1],deriv,'.')
				
				
			Ai=LP2.Aequalities[i,:]
			c_bar=LP2.costsvector+y_eq*LP2.Aequalities+y_ineq*LP2.Ainequalities
			alpha_optim=exactCoordinateLineSearch(Ai,LP2.Bequalities[i],c_bar)
			y_eq[i]+=alpha_optim
		for i in range(y_ineq.size):
			if False:
				y2=y_ineq.copy()
				vals=[]
				alphasgrid=np.linspace(-0.0116834,0.0039883,1000)
				for alpha in alphasgrid:
					y2[i]=y_ineq[i]+alpha	
					vals.append(eval(y_eq,y2))
				plt.plot(alphasgrid,vals,'.')
				deriv=np.diff(vals)/np.diff(alphasgrid)
				plt.plot(alphasgrid[:-1],deriv,'.')			
			Ai=LP2.Ainequalities[i,:]
			#c_bar=LP2.costsvector+y_eq*LP2.Aequalities+y_ineq*LP2.Ainequalities
			alpha_optim=exactCoordinateLineSearch(Ai,LP2.B_upper[i],c_bar)
			#prev_energy=eval(y_eq,y_ineq)
			prev_y_ineq=y_ineq[i]
			y_ineq[i]+=alpha_optim
			y_ineq[i]=max(y_ineq[i],0)
			diff_y_ineq=y_ineq[i]-prev_y_ineq			
			c_bar[Ai.indices]+=diff_y_ineq*Ai.data
			#new_energy=eval(y_eq,y_ineq)
			#assert(new_energy>=prev_energy-1e-5)
			#assert(np.max(y_ineq)<=0)
			
		c_bar,x=getOptimX(y_eq, y_ineq)
		max_violation=np.max(LP2.Ainequalities*x-LP2.B_upper)
		sum_violation= np.sum(np.maximum(LP2.Ainequalities*x-LP2.B_upper,0))
		E=eval(y_eq,y_ineq)
		print 'iter %d energy %f max violation %f sum_violation %f'%(iter ,E,max_violation,sum_violation)	
		if not callbackFunc is None:
			callbackFunc(0,x,0,0,0,0,0)
	
		#direction=scipy.sparse.csr.csr_matrix((y_ineq-y_ineq_prev))
		#coef_length=exactDualLineSearch(direction,LP2.Ainequalities,LP2.B_upper,c_bar,LP2.upperbounds,LP2.lowerbounds)
		#y_ineq=np.array(y_ineq_prev+coef_length*direction).flatten()	
		#y_ineq=np.maximum(y_ineq, 0)		
		#y_ineq=y_ineq+*0.1	
		#y_ineq=np.maximum(y_ineq, 0)
		#print 'iter %d energy %f'%(iter ,eval(y_eq,y_ineq))
	return x, y_eq,y_ineq


def primalCoordinateDescent(x,LP,nbmaxiter=1000,callbackFunc=None, constraints_coef=1000):
	# minimizes directly the linear program with a finit penalization on violated constraints
	# using coordinate descent
	#i.e. minimizes c'X + constraints_coef*sum(abs(Aeq*x-beq)) + constraints_coef*sum(maximum(Aineq*x-bineq))
	pass
	
		
def exactDualLineSearch(direction,A,b,c_bar,upperbounds,lowerbounds):

	assert(isinstance(direction,scipy.sparse.csr.csr_matrix))
	dA=direction*A		
	alphas=-c_bar[dA.indices]/dA.data
	order=np.argsort(alphas)
	dAU=dA.data*upperbounds[dA.indices]
	dAL=dA.data*lowerbounds[dA.indices]
	tmp1=np.minimum(dAU[order], dAL[order])
	tmp2=np.maximum(dAU[order], dAL[order])
	tmp3=np.cumsum(tmp2[::-1])[::-1]
	tmp4=np.cumsum(tmp1)
	derivatives=-(direction.dot(b))*np.ones(alphas.size+1)
	derivatives[:-1]+=tmp3
	derivatives[1:]+=tmp4

	#tmp=np.abs(Ai.data[order])*(LP2.lowerbounds[Ai.indices[order]]-LP2.upperbounds[Ai.indices[order]])
	#derivatives= -LP2.Bequalities[i]-np.sum(AiL[Ai.data>0])-np.sum(AiU[Ai.data<0])\
		#+np.hstack(([0],np.cumsum(tmp)))

	k=np.searchsorted(-derivatives,0)
	if derivatives[k]==0:
		t=np.random.rand()
		alpha_optim=t*alphas[order[k]]+(1-t)*alphas[order[k-1]]#maybe courld draw and random valu in the interval ? 
	else:
		alpha_optim=alphas[order[k-1]]	
	return alpha_optim


	
def dualGradientAscent(x,LP,nbmaxiter=1000,callbackFunc=None,y_eq=None,y_ineq=None):
	"""gradient ascent in the dual
	"""
	# convert to slack form (augmented form)
	LP2=copy.deepcopy(LP)
	LP=None
	#LP2.convertToSlackForm()
	assert(LP2.B_lower is None) or np.max(LP2.B_lower)==-np.inf
	#y_ineq=None
	#LP2.convertTo
	#LP2.convertToOnesideInequalitySystem()
	#LP2.upperbounds=np.minimum(10000,LP2.upperbounds)
	#LP2.lowerbounds=np.maximum(-10000,LP2.lowerbounds)
	#y0=None
	if y_eq is None:
		y_eq=np.zeros(LP2.Aequalities.shape[0])
		y_eq=-np.random.rand(y_eq.size)
	else:
		y_eq=y_eq.copy()
	#y_ineq=None
	if  y_ineq is None:
		if not(LP2.Ainequalities is None) :
			y_ineq=np.zeros(LP2.Ainequalities.shape[0])
			y_ineq=np.abs(np.random.rand(y_ineq.size))
	else:
		y_ineq=y_ineq.copy()
	#assert (LP2.B_lower is None)
	
	def getOptimX(y_eq,y_ineq):
		c_bar=LP2.costsvector.copy()
		if not LP2.Aequalities is None:
			c_bar+=y_eq*LP2.Aequalities
		if not LP2.Ainequalities is None:
			c_bar+=y_ineq*LP2.Ainequalities
		x=np.zeros(LP2.costsvector.size)
		x[c_bar>0]=LP2.lowerbounds[c_bar>0]
		x[c_bar<0]=LP2.upperbounds[c_bar<0]
		x[c_bar==0]=0.5*(LP2.lowerbounds+LP2.upperbounds)[c_bar==0]
		#t=np.random.rand(np.sum(c_bar==0))
		#x[c_bar==0]=t*LP2.lowerbounds[c_bar==0]+(1-t)*LP2.upperbounds[c_bar==0]
		
		return c_bar,x

	def eval(y_eq,y_ineq):
		c_bar,x=getOptimX(y_eq,y_ineq)
		
		#E=-y_eq.dot(LP2.Bequalities)-y_ineq.dot(LP2.B_upper)+np.sum(x*c_bar)
		#LP2.costsvector.dot(x)+y_ineq.dot(LP2.Ainequalities*x-LP2.B_upper)
		E=np.sum(np.minimum(c_bar*LP2.upperbounds,c_bar*LP2.lowerbounds)[c_bar!=0])
		if not LP2.Aequalities is None:
			E-=y_eq.dot(LP2.Bequalities)
		if not LP2.Ainequalities is None:
			E-=y_ineq.dot(LP2.B_upper)
		return E


	#x[c_bar==0]=0.5

	# alpha_i= vector containing the step lenghts that lead to a sign change on any of the gradient component 
	# when incrementing y[i]
	#
	print 'iter %d energy %f'%(0 ,eval(y_eq,y_ineq))
	
	
	
	prevE=eval(y_eq,y_ineq)
	for iter in range(nbmaxiter):
		c_bar,x=getOptimX(y_eq,y_ineq)
		if not LP2.Ainequalities is None:
			y_ineq_prev=y_ineq.copy()		
			max_violation=np.max(LP2.Ainequalities*x-LP2.B_upper)
			sum_violation=np.sum(np.maximum(LP2.Ainequalities*x-LP2.B_upper,0))
			np.sum(np.maximum(LP2.Ainequalities*x-LP2.B_upper,0))
			print 'iter %d energy %f max violation %f sum_violation %f'%(iter ,prevE,max_violation,sum_violation)		
	
			grad_y_ineq=LP2.Ainequalities*x-LP2.B_upper
			grad_y_ineq[y_ineq_prev<=0]=np.maximum(grad_y_ineq[y_ineq_prev<=0], 0)# not sure it is correct to do that here
			grad_y_ineq_sparse=scipy.sparse.csr.csr_matrix(grad_y_ineq)
			coef_length_ineq=exactDualLineSearch(grad_y_ineq_sparse,LP2.Ainequalities,LP2.B_upper,c_bar,LP2.upperbounds,LP2.lowerbounds)
			#y_ineq_prev+coef_length*grad_y>0
			assert(coef_length_ineq>=0)
			maxstep_ineq=np.min(y_ineq_prev[grad_y_ineq<0]/-grad_y_ineq[grad_y_ineq<0])
			coef_length_ineq=min(coef_length_ineq,maxstep_ineq)
			#if False:
				#y2=y_ineq.copy()
				#alphasgrid=np.linspace(coef_length*0.99,coef_length*1.01,1000)
				#vals=[]
				#for alpha in alphasgrid:
					#y2=y_ineq+alpha*grad_y		
					#vals.append(eval(y_eq,y2))
				#plt.plot(alphasgrid,vals,'.')		
			
			#coef_length=0.001/(iter+2000000)
			#coef_length=min(0.01/(iter+200000),maxstep)
			y_ineq=y_ineq_prev+coef_length_ineq*grad_y_ineq	
			#assert(np.min(y_ineq)>=-1e-8)
			y_ineq=np.maximum(y_ineq, 0)
			
		if not LP2.Aequalities is None and LP2.Aequalities.shape[0]>0:
			
			y_eq_prev=y_eq.copy()		
			max_violation=np.max(np.abs(LP2.Aequalities*x-LP2.Bequalities))
			sum_violation=np.sum(np.abs(LP2.Aequalities*x-LP2.Bequalities))
			
			print 'iter %d energy %f max violation %f sum_violation %f'%(iter ,prevE,max_violation,sum_violation)		
		
			grad_y_eq=LP2.Aequalities*x-LP2.Bequalities
			
			grad_y_eq_sparse=scipy.sparse.csr.csr_matrix(grad_y_eq)
			coef_length_eq=exactDualLineSearch(grad_y_eq_sparse,LP2.Aequalities,LP2.Bequalities,c_bar,LP2.upperbounds,LP2.lowerbounds)
			#y_ineq_prev+coef_length*grad_y>0
			assert(coef_length_eq>=0)
			
			y_eq=y_eq_prev+coef_length_eq*grad_y_eq	
					
		
		#while True:
			#y_ineq=y_ineq_prev+coef_length*grad_y		
			#newE=eval(y_eq,y_ineq)			
			#if newE< prevE:
				#coef_length=coef_length*0.5
				#print 'reducing step lenght'
			#else:
				#coef_length=coef_length*1.5
				#break
		newE=eval(y_eq,y_ineq)	
		prevE=newE
		if not callbackFunc is None and iter%100==0:
			callbackFunc(0,x,0,0,0,0,0)		

	print 'done'
	return x, y_eq,y_ineq




	
	
	
	
	


#@profile
def LP_primalDual(LP,simplex=None,x0=None,\
                  alpha=1,theta=1,nb_iter=100,callbackFunc=None,max_time=None,\
                  nb_iter_plot=300,save_problem=False,
                  frequency_update_active_set=20
                  ):
	
	# method adapted from 
	# Diagonal preconditioning for first order primal-dual algorithms in convex optimization 
	# by Thomas Pack and Antonin Chambolle
	# the adaptatio makes the code able to handle a more flexible specification of the LP problem
	# (we could transform genric LPs into the equality form , but i am note sure the convergence would be the same)
	# minimizes c.T*x 
	# such that 
	# Aeq*x=beq
	# b_lower<= Aineq*x<= b_upper               assert(scipy.sparse.issparse(Aineq))

	# lb<=x<=ub
	#callbackFunc=None
	
	
	LP2=copy.deepcopy(LP)
	LP2.convertToOnesideInequalitySystem()
	
	LP2.upperbounds=np.minimum(10000,LP2.upperbounds)
	LP2.lowerbounds=np.maximum(-10000,LP2.lowerbounds)	
	
	c=LP2.costsvector
	Aeq=LP2.Aequalities
	if Aeq.shape[0]==0:
		Aeq=None
		y_eq=None
	beq=LP2.Bequalities
	Aineq=LP2.Ainequalities
	b_lower=LP2.B_lower
	b_upper=LP2.B_upper
	bineq=b_upper
	lb=LP2.lowerbounds
	ub=LP2.upperbounds
	#c,Aeq,beq,Aineq,b_lower,b_upper,lb,ub
	assert(b_lower is None)
	assert(lb.size==c.size)
	assert(ub.size==c.size)	
	
	start = time.clock() 				
	elapsed=start
	
	
	
	
		

	use_vec_sparsity=False
	if x0!=None:
		x=xo.copy()
	else:
		x=np.zeros(c.size)


	#save_problem=True
	if save_problem:
		with open('LP_problem2.pkl', 'wb') as f:
			d={'c':c,'Aeq':Aeq,'beq':beq,'Aineq':Aineq,'bineq':bineq,'lb':lb,'ub':ub}	
			import pickle
			pickle.dump(d,f)	

	n=c.size		
	
	useStandardForm=False
	if useStandardForm:
		c,Aeq,beq,lb,ub,x0=convertToStandardFormWithBounds(c,Aeq,beq,Aineq,bineq,lb,ub,x0)
		Aineq=None





	useColumnPreconditioning=True
	if useColumnPreconditioning:
		# constructing the preconditioning diagonal matrices  
		tmp=0
		if Aeq!=None:
			print "Aeq shape="+str(Aeq.shape)
	
			assert(scipy.sparse.issparse(Aeq))  
			assert(Aeq.shape[1]==c.size)
			assert(Aeq.shape[0]==beq.size)
			AeqCopy=Aeq.copy()               
			AeqCopy.data=np.abs(AeqCopy.data)**(2-alpha)        
			SumAeq=(np.ones((1,AeqCopy.shape[0]))*AeqCopy)  
			tmp=tmp+SumAeq
			#AeqT=Aeq.T
		if Aineq!=None:     
			print "Aineq shape="+str(Aineq.shape)
			assert(scipy.sparse.issparse(Aineq))
			assert(Aineq.shape[1]==c.size)
			assert(Aineq.shape[0]==b_upper.size)
			AineqCopy=Aineq.copy()               
			AineqCopy.data=np.abs(AineqCopy.data)**(2-alpha)        
			SumAineq=(np.ones((1,AineqCopy.shape[0]))*AineqCopy)   
			tmp=tmp+ SumAineq
			#AineqT=Aineq.T
		if Aeq==None and Aineq==None:
			x=np.zeros_like(lb)
			x[c>0]=lb[c>0]
			x[c<0]=ub[c<0]                
			return x
		tmp[tmp==0]=1
		diagT=1/tmp[0,:]
		T=scipy.sparse.diags(diagT[None,:],[0]) .tocsr()   
	else:
		scipy.sparse.eye(len(x))  
		diagT=np.ones(x.shape)
		
		
	if Aeq!=None:
		AeqCopy=Aeq.copy()
		AeqCopy.data=np.abs(AeqCopy.data)**(alpha)
		SumAeq=AeqCopy*np.ones((AeqCopy.shape[1]))
		tmp=SumAeq
		tmp[tmp==0]=1
		diagSigma_eq=1/tmp
		Sigma_eq=scipy.sparse.diags([diagSigma_eq],[0]).tocsr()
		y_eq=np.zeros(Aeq.shape[0])
		del AeqCopy
		del SumAeq
	if Aineq!=None:  
		AineqCopy=Aineq.copy()
		AineqCopy.data=np.abs(AineqCopy.data)**(alpha)
		SumAineq=AineqCopy*np.ones((AineqCopy.shape[1])) 
		tmp=SumAineq
		tmp[tmp==0]=1
		diagSigma_ineq=1/tmp
		Sigma_ineq=scipy.sparse.diags([diagSigma_ineq],[0]).tocsr()     
		y_ineq=np.zeros(Aineq.shape[0])
		del AineqCopy
		del SumAineq

	#some cleaning 
	del tmp


	#del diagSigma
	#del diagT



	#iterations        
	#AeqT=AeqT.tocsc()callbackFunc
	#AineqT=AineqT.tocsc()
	x3=x
	
	best_integer_solution_energy=np.inf
	best_integer_solution=None
	
	list_active_variables=np.arange(x.size)
	is_active_variable=np.ones(x.shape,dtype=np.bool)
	if not Aineq is None:
		is_active_inequality_constraint=np.ones(Aineq.shape[0],dtype=np.bool)
		list_active_inequality_constraints=np.arange(Aineq.shape[0])
	if not Aeq is None:
		is_active_equality_constraint=np.ones(Aeq.shape[0],dtype=np.bool)
		list_active_equality_constraints=np.arange(Aeq.shape[0])
		AeqCSC=Aeq.tocsc()
		AeqCSR=Aeq.tocsr()
		r_eq=(Aeq*x)-beq
		r_eq_active=r_eq[list_active_equality_constraints]
		subAeqCSR=AeqCSR[list_active_equality_constraints,:]
		subAeqCSC2=subAeqCSR.tocsc()[:,list_active_variables]
		
		subAeqCSR2=subAeqCSC2.tocsr()		
	d=c.copy()
	if not Aineq is None:
		AineqCSC=Aineq.tocsc()	
		AineqCSR=Aineq.tocsr()	
		r_ineq=(Aineq*x)-bineq		
		#subAeqCSC=AeqCSC[list_active_equality_constraints,:]	
		#subAineqCSC=AineqCSC[list_active_inequality_constraints,:]
		r_ineq_active=r_ineq[list_active_inequality_constraints]
		subAineqCSR=AineqCSR[list_active_inequality_constraints,:]	
		subAineqCSC2=subAineqCSR.tocsc()[:,list_active_variables]
		subAineqCSR2=subAineqCSC2.tocsr()	
		diagSigma_ineq_active=diagSigma_ineq[list_active_inequality_constraints]
		active_y_ineq=y_ineq[list_active_inequality_constraints]
		
	x_active=x[list_active_variables]
	lb_active=lb[list_active_variables]
	ub_active=ub[list_active_variables]	
	
	
	
	diagT_active=diagT[list_active_variables]
	d_active=d[list_active_variables]

	x3=x
	x3_active=x3[list_active_variables]
	for i in range(nb_iter):
		
		#Update he primal variables
		
		if Aeq!=None:				
			if use_vec_sparsity:
				yeq_sparse=scipy.sparse.coo_matrix(y_eq).T
				d=d+(yeq_sparse*Aeq).toarray().ravel()#faster when few constraint are activated
			else:
				if i==0:
					#d=d+y_eq*Aeq
					d_active=d_active+y_eq*subAeqCSR2
				else:
					#sparse_diff_y_eq=scipy.sparse.csr_matrix((diff_active_y_eq, list_active_equality_constraints, [0,list_active_equality_constraints.size]), shape=(1,Aeq.shape[0]))
					#d=d+diff_active_y_eq*Aeq[list_active_equality_constraints,:]
					
					#increment=sparse_diff_y_eq*AeqCSR
					#d=d+increment.toarray().ravel()
					#d[increment.indices]=d[increment.indices]+increment.data#  does numpoy exploit the fact that second term is sparse ?
					#d=d+diff_active_y_eq*subAeqCSR
					d_active=d_active+diff_active_y_eq*subAeqCSR2 #will be usefull when few active variables
				#d+=y_eq*Aeq# strangley this does not work, give wrong results
				

		if Aineq!=None: 			
			if use_vec_sparsity:
				yineq_sparse=scipy.sparse.coo_matrix(y_ineq).T
				d=d+(yineq_sparse*Aineq).toarray().ravel()#faster when few constraint are activated
			else:
				if i==0:
					#d=d+y_ineq*Aineq
					d_active=d_active+y_ineq*subAineqCSR2
				else:

					
					#d=d+diff_active_y_ineq*Aineq[list_active_inequality_constraints,:]# does numpoy exploit the fact that second term is sparse ?
					#sparse_diff_y_ineq=scipy.sparse.csr_matrix((diff_active_y_ineq, list_active_inequality_constraints, [0,list_active_inequality_constraints.size]), shape=(1,Aineq.shape[0]))
					#increment=sparse_diff_y_ineq*AineqCSR
					#d=d+increment.toarray().ravel()	
					#d[increment.indices]=d[increment.indices]+increment.data
					#d=d+diff_active_y_ineq*subAineqCSR
					d_active=d_active+diff_active_y_ineq*subAineqCSR2
				#d+=y_ineq*Aineq
				
		# update list active variables
		# ideally find largest values in primal steps diagT*d, should ne do that at each step, or have a
		# structure that allows to maintaint a list of larges values
		if True and i>0 and i%frequency_update_active_set==0:
			x[list_active_variables]=x_active
			x3[list_active_variables]=x3_active
			y_ineq[list_active_inequality_constraints]=active_y_ineq
			# update d for the variables that where inactive
			
			d=c+y_ineq*AineqCSR
			if Aeq!=None:
				d+=y_eq*AeqCSR
			
			#tmp=np.minimum(x-lb,np.maximum(diagT*d,0))+np.minimum(ub-x,np.maximum(-diagT*d,0))
			
			if True:
				tmp=np.abs(diagT*d)
				is_active_variable=(tmp>1e-6) 
				is_active_variable[ (diagT*d>1e-3) & (x-lb==0)]=False
				is_active_variable[ (diagT*d<-1e-3) & (ub-x==0)]=False
				list_active_variables,=np.nonzero(is_active_variable)# garde les 10 % le plus larges
			d_active=d[list_active_variables]
			
			# update list active constraints
			# ideally find largest values in duals steps diagSigma_ineq*r_ineq
			if Aeq!=None:
				r_eq=(AeqCSC*x3)-beq
			r_ineq=(AineqCSC*x3)-bineq	
			#tmp=np.abs(diagSigma_ineq*r_ineq)*((y_ineq>0) | (diagSigma_ineq*r_ineq>0 ))
			#list_active_inequality_constraints,=np.nonzero(tmp>np.percentile(tmp, 10))
			is_active_inequality_constraint= (r_ineq>-0.2) | (y_ineq>0)
			list_active_inequality_constraints,=np.nonzero(is_active_inequality_constraint)
			
			if Aeq!=None:
				tmp=np.abs(diagSigma_eq*r_eq)
				is_active_equality_constraint=tmp>1e-6
				list_active_equality_constraints,=np.nonzero(is_active_equality_constraint)
				nb_active_equality_constraints=np.sum(is_active_equality_constraint)
				percent_active_equality_constraint=100*np.mean(is_active_equality_constraint)
			else:
				nb_active_equality_constraints=0
				percent_active_equality_constraint=0
			
			#subAeqCSC=AeqCSC[list_active_equality_constraints,:]
			#subAeqCSC=AeqCSR[list_active_equality_constraints,:].tocsc()
			if Aeq!=None:
				r_eq_active=r_eq[list_active_equality_constraints]
			#subAineqCSC=AineqCSC[list_active_inequality_constraints,:]
			#subAineqCSC=AineqCSR[list_active_inequality_constraints,:].tocsc()
			if Aeq!=None:
				subAeqCSR=AeqCSR[list_active_equality_constraints,:]
				subAeqCSC2=subAeqCSR.tocsc()[:,list_active_variables]
				subAeqCSR2=subAeqCSC2.tocsr()
			subAineqCSR=AineqCSR[list_active_inequality_constraints,:]
			
			
			subAineqCSC2=subAineqCSR.tocsc()[:,list_active_variables]
			
			subAineqCSR2=subAineqCSC2.tocsr()
			r_ineq_active=r_ineq[list_active_inequality_constraints]				
			print '%d active variables %d  active inequalities %d active equalities'%(np.sum(is_active_variable),np.sum(is_active_inequality_constraint),nb_active_equality_constraints)
			
			print '%f percent of active variables %f percent active inequalities %f percent active equalities'%(100*np.mean(is_active_variable),100*np.mean(is_active_inequality_constraint),percent_active_equality_constraint)
			x_active=x[list_active_variables]
			x3_active=x3[list_active_variables]
			lb_active=lb[list_active_variables]
			ub_active=ub[list_active_variables]
			diagT_active=diagT[list_active_variables]
			diagSigma_ineq_active=diagSigma_ineq[list_active_inequality_constraints]
			active_y_ineq=y_ineq[list_active_inequality_constraints]
		#x2=x-T*d
		
		
		
		
		
		new_active_x=x_active-diagT_active*d_active
		#np.maximum(x2,lb,x2)
		#np.minimum(x2,ub,x2)

		if not simplex is None:
			
			new_active_x.flat[simplex]=projsplx(new_active_x[simplex])
		np.maximum(new_active_x,lb_active,new_active_x)
		np.minimum(new_active_x,ub_active,new_active_x)
		
		x3_prev=x3_active
		x3_active=(1+theta)*new_active_x-theta*x_active # smoothing ?
		#diff_x3=x3_prev-x3
		diff_active_x3=x3_active-x3_prev
		x_active=new_active_x
		#sparse_diff_x=scipy.sparse.csc_matrix((diff_active_x, list_active_variables, [0,list_active_variables.size]), shape=(x.size,1))

		
		if use_vec_sparsity:
			x3_sparse=scipy.sparse.coo_matrix(x3).T
		if Aeq!=None:
			if use_vec_sparsity:
				r_eq=(Aeq*x3_sparse).toarray().ravel()-beq
			else:	
				
				#r_eq=r_eq+(Aeq[:,list_active_variables]*diff_active_x)# can use sparisity in diff_x3
				#r_eq=r_eq+Aeq*sparse_diff_x
				#increment=subAeqCSC*sparse_diff_x # to do : update only the active constraints residuals
				#r_eq_active[increment.indices]=r_eq_active[increment.indices]+increment.data 
				
				
				r_eq_active+=subAeqCSC2*diff_active_x3
				#r_eq=r_eq+increment.toarray().ravel()
						
		if Aineq!=None: 			
			if use_vec_sparsity:
				r_ineq=(Aineq*x3_sparse).toarray().ravel()-bineq
			else:
			
				#r_ineq=r_ineq+(Aineq[:,list_active_variables]*diff_active_x)
				#r_ineq=r_ineq+Aineq*sparse_diff_x
				#increment=subAineqCSC*sparse_diff_x# to do : update only the active constraints residuals
				#increment=subAineqCSC2*diff_active_x
				#r_ineq=r_ineq+increment.toarray().ravel()
				#r_ineq_active[increment.indices]=r_ineq_active[increment.indices]+increment.data 
				r_ineq_active+=subAineqCSC2*diff_active_x3
		
		if i> 0 and i%nb_iter_plot==0:
			x[list_active_variables]=x_active
			if not Aineq is None:
				y_ineq[list_active_inequality_constraints]=active_y_ineq
				r_ineq=(AineqCSC*x)-bineq
			if Aeq!=None:
				r_eq=(AeqCSC*x)-beq
			
			
			prev_elapsed=elapsed
			elapsed= (time.clock() - start)	
			mean_iter_priod=(elapsed-prev_elapsed)/10
			if max_time!=None and elapsed>max_time:
				break			
			energy1=c.dot(x)
			
			# x4 is obtained my minimizing with respect to the primal variable while keeping the langrangian coef fix , which give a lower bound on the optimal solution
			# energy2 is the lower bound
			# energy1  is the value of the lagrangian at the curretn (hopefull sadle) point
			x4=np.zeros(lb.size)
		
			#c_bar=LP2.costsvector.copy()
			#if not LP2.Aequalities	 is None and LP2.Aequalities.shape[0]>0:
				#c_bar+=y_eq*LP2.Aequalities
			#if not LP2.Ainequalities is None:
				#c_bar+=y_ineq*LP2.Ainequalities	
			
			d[list_active_variables]=d_active
			c_bar=d.copy()	
			x4[c_bar>0]=lb[c_bar>0]
			x4[c_bar<0]=ub[c_bar<0]
			x4[c_bar==0]=0
			
			
			energy2=np.sum(x4*c_bar)
			
			max_violated_equality=0
			max_violated_inequality=0
			if Aeq!=None:
				energy1+=y_eq.T.dot(Aeq*x-beq)
				energy2-=y_eq.dot(beq)
				max_violated_equality=np.max(np.abs(r_eq))
			if Aineq!=None:    
				energy1+=y_ineq.T.dot(Aineq*x-bineq)
				energy2-=y_ineq.dot(bineq)
				max_violated_inequality=np.max(r_ineq)
				
		

			xrounded=np.round(x)
			#xrounded=greedy_round(x,c,Aeq,beq,Aineq,np.full(bineq.shape,-np.inf),bineq,lb.copy(),ub.copy(),callbackFunc=callbackFunc)
			
			energy_rounded=c.dot(xrounded)
			if Aeq!=None:
				nb_violated_equality_rounded=np.sum(np.abs(Aeq*xrounded-beq))
			else:
				nb_violated_equality_rounded=0
			if Aineq!=None:	
				nb_violated_inequality_rounded=np.sum(np.maximum(Aineq*xrounded-bineq,0))	
			else:
				nb_violated_inequality_rounded=0
			
			if nb_violated_equality_rounded==0 and nb_violated_inequality_rounded==0:
				print '##########   found feasible solution with energy'+str(energy_rounded)
				if energy_rounded<best_integer_solution_energy:
					best_integer_solution_energy=energy_rounded
					best_integer_solution=xrounded
				

			print 'iter'+str(i)+": energy1= "+str(energy1) + " energy2="+str(energy2)+ ' elaspsed '+str(elapsed)+' second'+\
				  ' max violated inequality:'+str(max_violated_inequality)+\
				  ' max violated equality:'+str(max_violated_equality)+\
				  'mean_iter_period='+str(mean_iter_priod)+\
			'rounded : %f ineq %f eq'%(nb_violated_inequality_rounded,nb_violated_equality_rounded)
				#'y_eq has '+str(100 * np.mean(y_eq==0))+' % of zeros '+\
		#    'y_ineq has '+str(100 * np.mean(y_ineq==0))+' % of zeros '+\			 
		
			if callbackFunc!=None:

				callbackFunc(i,x,energy1,energy2,elapsed,max_violated_equality,max_violated_inequality,is_active_variable=is_active_variable)

		#Update the dual variables
	
		if Aeq!=None:
			diff_active_y_eq=diagSigma_eq[list_active_equality_constraints]*r_eq_active
			y_eq[list_active_equality_constraints]=y_eq[list_active_equality_constraints]+diff_active_y_eq
			
			#y_eq=y_eq+diagSigma_eq*r_eq
			#y_eq+=diagSigma_eq*r_eq

		if Aineq!=None: 
			#active_y_ineq=y_ineq[list_active_inequality_constraints]
			new_active_y_ineq=active_y_ineq+diagSigma_ineq_active*r_ineq_active
			new_active_y_ineq=np.maximum(new_active_y_ineq, 0)			
			diff_active_y_ineq=new_active_y_ineq-active_y_ineq
			active_y_ineq=new_active_y_ineq# np.mean(diff_active_y_ineq!=0) often give me 0.05 on the facade , can i use that for more speedups ?
			#y_ineq[list_active_inequality_constraints]=active_y_ineq
			
			#y_ineq+=diagSigma_ineq*r_ineq			
			#np.maximum(y_ineq, 0,y_ineq) 
	
	#best_integer_solution, y_eq,y_ineq=dualGradientAscent(x,LP2,callbackFunc=callbackFunc,y_eq=y_eq,y_ineq=y_ineq,nbmaxiter=10000)
			
	#best_integer_solution, y_eq,y_ineq=dualCoordinateAscent(x,LP2,callbackFunc=callbackFunc,y_eq=y_eq,y_ineq=y_ineq,nbmaxiter=100)
			
	best_integer_solution=greedy_fix(best_integer_solution,LP,callbackFunc=callbackFunc)
	#best_integer_solution=dualCoordinateAscent(x,LP,callbackFunc=callbackFunc,y_eq=y_eq,y_ineq=y_ineq)
	#best_integer_solution, y_eq,y_ineq=dualGradientAscent(x,LP,callbackFunc=callbackFunc,y_eq=y_eq,y_ineq=y_ineq,nbmaxiter=10)
	#best_integer_solution, y_eq,y_ineq=dualGradientAscent(x,LP2,callbackFunc=callbackFunc,y_eq=y_eq,y_ineq=y_ineq,nbmaxiter=10)
	#best_integer_solution=greedy_round(x,LP,callbackFunc=callbackFunc)
	if not best_integer_solution is None:
		best_integer_solution=best_integer_solution[:n]
		callbackFunc(0,best_integer_solution,0,0,0,0,0)
		
	return x[:n],best_integer_solution


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
		if self.Aeq!=None:							
			max_violated_equality=np.max(np.abs(r_eq))
		if self.Aineq!=None: 
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
	
		

def LP_primalDualCondat(c,Aeq,beq,Aineq,b_lower,b_upper,lb,ub,simplex=None,\
                        x0=None,alpha=1,theta=1,nb_iter=100,callbackFunc=None,\
                        max_time=None,save_problem=False,useStandardForm=False):
	# minimizes c.T*x 
	# such that 
	# Aeq*x=beq
	# b_lower<= Aineq*x<= b_upper 
	# lb<=x<=ub
	#
	# method adapted from 
	# A Generic Proximal Algorithm for Convex Optimization
	# Application to Total Variation Minimization	


	Aineq,bineq=convertToOnesideInequalitySytem(Aineq,b_lower,b_upper)

	if x0!=None:
		x=xo.copy()
	else:
		x=np.zeros(c.size)
	assert(lb.size==c.size)
	assert(ub.size==c.size)

	#save_problem=True
	if save_problem:
		with open('LP_problem2.pkl', 'wb') as f:
			d={'c':c,'Aeq':Aeq,'beq':beq,'Aineq':Aineq,'bineq':bineq,'lb':lb,'ub':ub}	
			import pickle
			pickle.dump(d,f)	

	n=c.size		
	
	if useStandardForm and Aineq!=None:
		c,Aeq,beq,lb,ub,x0=convertToStandardFormWithBounds(c,Aeq,beq,Aineq,bineq,lb,ub,x0)
		Aineq=None
		
	st=solutionStat(c,AeqCSC,beq,AineqCSC,bineq,callbackFunc)

	for i in range(nb_iter):

		#Update the primal variables

		d=c+y_eq*AeqCSR+y_ineq*AineqCSR
		xtilde=x-tau*d		
		if not simplex is None:
			x.flat[simplex]=projsplx(xtilde[simplex])
		np.maximum(xtilde,lb,xtilde)
		np.minimum(xtilde,ub,xtilde)
		z=2*xtilde-x
		x=rho*tilde+(1-rho)*x
		

		if i%30==0:
			st.eval(x, i)
			if max_time!=None and st.elapsed>max_time:
				break
			
		r_eq=(AeqCSC*z)-beq
		r_ineq=(AineqCSC*z)-bineq
		
		y_eq_tilde=y_eq+sigma*r_eq	
		y_eq=rho*y_eq_tilde+(1-rho)*y_eq
		y_ineq_tilde=y_ineq+sigma*r_ineq
		y_ineq_tilde=np.maximum(y_ineq_tilde, 0)
		y_ineq=rho*y_ineq_tilde+(1-rho)*y_ineq

	if not best_integer_solution is None:
		best_integer_solution=best_integer_solution[:n]
	return x[:n],best_integer_solution

def test():



	yeq_sparse=scipy.sparse.coo_matrix(y_eq)

	de={}
	de['full']=y_eq
	#de['csr']=yeq_sparse.tocsr()
	#de['csc']=yeq_sparse.tocsc()
	#de['lil']=yeq_sparse.tolil()
	#de['dok']=yeq_sparse.todok()	

	da={}
	#da['csr']=Aeq.tocsr()
	#da['csc']=Aeq.tocsc()
	#da['coo']=Aeq.tocoo()
	#da['lil']=Aeq.tolil()
	#da['dok']=Aeq.todok()
	da['bsr']=Aeq.tobsr()





	for keyY,y in de.items():
		for keyA,A in da.items():
			#if keyT=='dok' and 
			print 'y_'+keyY+ '* A_'+keyA+ ' takes...',
			start = time.clock() 
			for i in range(20):
				r=y*A

			elapsed= (time.clock() - start)
			print elapsed


			#y_lil* A_lil takes... 49.67
			#y_lil* A_bsr takes... 9.05
			#y_lil* A_coo takes... 4.06
			#y_lil* A_csc takes... 4.2
			#y_lil* A_csr takes... 1.94
			#y_dok* A_lil takes... 49.24
			#y_dok* A_bsr takes... 9.01
			#y_dok* A_coo takes... 4.14
			#y_dok* A_csc takes... 4.17
			#y_dok* A_csr takes... 1.93
			#y_csc* A_lil takes... 54.28
			#y_csc* A_bsr takes... 13.41
			#y_csc* A_coo takes... 8.44
			#y_csc* A_csc takes... 2.35
			#y_csc* A_csr takes... 6.82
			#y_csr* A_lil takes... 49.24
			#y_csr* A_bsr takes... 8.96
			#y_csr* A_coo takes... 4.03
			#y_csr* A_csc takes... 4.16
			#y_csr* A_csr takes... 1.95	
			#y_full* A_coo takes... 2.38
			#y_full* A_csc takes... 3.6
			#y_full* A_csr takes... 1.61


#A=Aeq_csr 
#r=np.zeros(A.shape[1],dtype=np.float)
#for i in range(len(y_eq)):
	#if y_eq[i]!=0:
		#for k in range(Aeq_csr.indptr[i],Aeq_csr.indptr[i+1]):
			#j=A.indices[k]
			#r[j]+=A.data[k]*y_eq[i]







if __name__ == "__main__":
	plt.ion()
	from constraints import *
	nLabels = 1  
	np.random.seed(1)
	size_image=(50,50,nLabels)   
	nb_pixels=size_image[0]*size_image[1]
	unary_terms=(np.random.rand(size_image[0],size_image[1],size_image[2]))*2-1
	#import mosekHelper
	#LP=LinearProgramMosek()
	LP=LinearProgramScipy()
	
	
	
	#LP=getLinearProgram()
	indices=LP.addVariablesArray(shape=size_image,lowerbounds=0,upperbounds=1,costs=unary_terms) 
	LP.addGlobalRowConvexityConstraint(indices,label=0)
	LP.addGlobalColumnConvexityConstraint(indices,label=0)
	#LP.addPottModel(indices, coefpenalization=0.2)
	#LP.addPottHorizontal(indices, coefpenalization=0.5)
	#LP.addGridConstraint(indices, label=0)
	print "solving"
	#LP2=copy.deepcopy(LP)
	#LP2.convertToSlackForm()
	#LP.upperbounds=np.minimum(100000,LP.upperbounds)# strangely putting large values cause mosek to find a non optimal solution...
	#LP.lowerbounds=np.maximum(-100000,LP.lowerbounds)
	LP.upperbounds=np.minimum(100,LP.upperbounds)
	LP.lowerbounds=np.maximum(-100,LP.lowerbounds)	
	
	#solution2=LP2.convertToMosek().solve()
	LP2=copy.deepcopy(LP)
	LP2.convertToOnesideInequalitySystem()
	#LP2.saveMPS('mosek.mps')
	solution=LP.convertToMosek().solve()
	#solution,elapsed=LP.solve(nb_iter=2000,method='ChambollePock',frequency_update_active_set=100000)
	
	#solution,elapsed=LP.solve(method=LP.optimizertype_intpnt ,force_integer=False,getTiming=True)
	image=solution[indices] 
	groundTruth=image
	energy_optimal_mosek=np.sum(image*unary_terms)	
	
	#new instance, to see if solving the same problem with new data term is faster (reuse of some anaysis of the constraint matrix, like preconditionning ? )
	#unary_terms=(np.random.rand(size_image[0],size_image[1],size_image[2]))*2-1
	#LP.setCostsVariables(indices,costs=unary_terms) 
	#solution,elapsed=LP.solve(method=LP.optimizertype_intpnt ,force_integer=False,getTiming=True)
	#image=solution[indices] 
	#groundTruth=image
	#energy_optimal_mosek=np.sum(image*unary_terms)	
	
	#import glpk 
	##glpk.glpk('mosek_task.mps')
	##task=glpk.glpk('mosek_task.lp')
	##task.solve()
	##d=task.solution()
	##ids=[int(k[1:]) for k in d.keys()]
	##solution=np.array(s.values())
	##image=solution[indices] 
	##lp = glpk.LPX(mps='mosek_task.mps')
	##lp = glpk.LPX(lp='mosek_task.lp')
	##lp = glpk.LPX(freemps='mosek_task.mps')
	
	##/usr/local/lib/python2.7/dist-packages/pulp/solverdir/cbc-64 mosek_task.mps branch printingOptions rows solution /tmp/19867-pulp.sol '
	#import pulp
	#lp=pulp.LpProblem('ppppp')
	
	#lp.solve(use_mps=False)
	##lp.solver.actualSolve(lp,use_mps=False)
	#pulp.solvers.GLPK_CMD().actualSolve(lp)
	#p.solver.actualSolve(lp,(),use_mps=False)
	#p.solver.actualSolve(pulp.solvers.PYGLPK(),use_mps=False)#AttributeError: PYGLPK instance has no attribute 'writeMPS'
	
	
	plt.imshow(image[:,:,0],cmap=plt.cm.Greys_r,interpolation='none')
	plt.imshow(groundTruth[:,:,0],cmap=plt.cm.Greys_r,interpolation='none')
	
	LP=LinearProgramScipy()
	indices=LP.addVariablesArray(shape=size_image,lowerbounds=0,upperbounds=1,costs=unary_terms) 
	LP.addGlobalRowConvexityConstraint(indices,label=0,name='globalRowConvexity')
	LP.addGlobalColumnConvexityConstraint(indices,label=0,name='globalColumnConvexity')
	#LP.addPottModel(indices, coefpenalization=0.2)
	
	#LP.addPottHorizontal(indices, coefpenalization=0.5)
	#LP.addGridConstraint(indices, label=0)
	fig_solutions=plt.figure()
	
	im=plt.imshow(unary_terms[:,:,0],cmap=plt.cm.Greys_r,interpolation="nearest",vmin=0,vmax=1)
	fig_curves=plt.figure()
	ax_curves1=plt.gca()	
	fig_curves=plt.figure()
	ax_curves2=plt.gca()		
	def plotSolution(niter,solution,is_active_variable=None):
		image=solution[indices] 
		imsave('../paper/images2/testLP/iter%05d.png'%niter,solution[indices][:,:,0])
		imsave('../paper/images2/testLP/diff_iter%05d.png'%niter,np.diff(solution[indices][:,:,0]))
		im.set_array(image[:,:,0])
		#im.set_array(np.diff(image[:,:,0]))	
		plt.draw()
		plt.show()

	sol2,elapsed=LP.solve(method="ChambollePock",force_integer=False,getTiming=True,\
	                      nb_iter=1000,nb_iter_plot=10,groundTruth=groundTruth,\
	                      groundTruthIndices=indices,plotSolution=plotSolution,\
	                      frequency_update_active_set=1000000)
	ax_curves1.plot(LP.itrn_curve,LP.distanceToGroundTruth,label='ChambollePock')	
	ax_curves2.plot(LP.dopttime_curve,LP.distanceToGroundTruth,label='ChambollePock')
	methods=["ADMM","ADMM2","ADMMBlocks"]
	for method in methods:
		#method=
		sol1,elapsed=LP.solve(method=method,force_integer=False,getTiming=True,nb_iter=1000,max_time=30,groundTruth=groundTruth,groundTruthIndices=indices,plotSolution=None)
		ax_curves1.plot(LP.itrn_curve,LP.distanceToGroundTruth,label=method)
		
		ax_curves2.plot(LP.dopttime_curve,LP.distanceToGroundTruth,label=method)
		ax_curves1.legend()
		ax_curves2.legend()
		plt.draw()
		plt.show()	
	#plt.figure()
	#plt.plot(LP.itrn_curve,LP.dopttime_curve,'g',label='ADMM')
	#plt.draw()
	#plt.show()	
	
	plt.figure()
	
	plt.subplot(1,3,1)
	plt.imshow(groundTruth[:,:,0],cmap=plt.cm.Greys_r,interpolation='none')
	plt.subplot(1,3,2)
	plt.imshow(sol1[indices][:,:,0],cmap=plt.cm.Greys_r,interpolation='none')
	plt.subplot(1,3,3)
	plt.imshow(sol2[indices][:,:,0],cmap=plt.cm.Greys_r,interpolation='none')	
	plt.ioff()
	
	#with open('LP_problem.pkl', 'rb') as f:

		#import pickle	
		#start = time.clock() 	
		#d=pickle.load(f)
		#elapsed= (time.clock() - start)
		#print 'took '+str(elapsed)+' seconds to load the LP problem'
		#LP_primalDual(d['c'],d['Aeq'],d['beq'],d['Aineq'],d['bineq'],d['lb'],d['ub'],max_time=100)
