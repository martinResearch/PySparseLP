import copy
import numpy as np
import time
import scipy.sparse
import scipy.ndimage


def DualCoordinateAscent(x,LP,nbmaxiter=20,callbackFunc=None,y_eq=None,y_ineq=None,max_time=None):
	"""Method from 'An algorigthm for large scale 0-1 integer 
	programming with application to airline crew scheduling'
	we generelized it to the case where A is not 0-1 and
	the upper bounds can be greater than 1
	did not generalize and code the approximation method
	"""
	start=time.clock()
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
		elapsed= (time.clock() - start)	
		if not callbackFunc is None:
			callbackFunc(iter,x,0,0,elapsed,0,0)
	
		#direction=scipy.sparse.csr.csr_matrix((y_ineq-y_ineq_prev))
		#coef_length=exactDualLineSearch(direction,LP2.Ainequalities,LP2.B_upper,c_bar,LP2.upperbounds,LP2.lowerbounds)
		#y_ineq=np.array(y_ineq_prev+coef_length*direction).flatten()	
		#y_ineq=np.maximum(y_ineq, 0)		
		#y_ineq=y_ineq+*0.1	
		#y_ineq=np.maximum(y_ineq, 0)
		#print 'iter %d energy %f'%(iter ,eval(y_eq,y_ineq))
		
		if (not max_time is None) and elapsed>max_time:
			break			
	return x, y_eq,y_ineq

