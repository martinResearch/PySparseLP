import copy
import numpy as np
import time
import scipy.sparse
import scipy.ndimage
from tools import preconditionConstraints,convertToStandardFormWithBounds  ,chrono,check_decrease
from gaussSiedel import boundedGaussSeidelClass
import  scikits.sparse.cholmod
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


