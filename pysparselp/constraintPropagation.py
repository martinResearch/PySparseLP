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



# These function implement a very naive methods to round the solution of a continuous 
# linear program solution to an integer solution using 
# - constraints propagation and backtracking
# - greedy reduction of the number of violated constraints using local search
import numpy as np
import copy



try:
	import cython_tools
	cython_tools_installed=True
except:
	print 'could not import cython_tools maybe the compilation did not work , will be slower'
	cython_tools_installed=False

def check_constraints(i,x_r,mask,Acsr,Acsc,b_lower,b_upper):
	"""check that the variable i is not involed in any violated constraint"""
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


#@profile	
def propagateConstraints(list_changed_var,x_l,x_u,Acsr,Acsc,b_lower,b_upper,back_ops,nb_iter=1000,use_cython=True):
	# may have similarities with the tightening method in http://www.davi.ws/doc/gondzio94presolve.pdf
	if cython_tools_installed and use_cython:
		#return cython_tools.propagateConstraints(list_changed_var,x_l,x_u,Acsr,Acsc,b_lower,b_upper,back_ops,nb_iter=nb_iter)
		return cython_tools.propagateConstraints(list_changed_var,x_l,x_u,\
		        Acsc.indices,\
			Acsr.indices,\
			Acsr.indptr,\
			Acsc.indptr,\
			Acsr.data,\
		        b_lower,b_upper,back_ops,nb_iter=nb_iter)
	tol=1e-5 # to cope with small errors
	
	for iter in range(nb_iter):
		#print '%d variable fixed '% np.sum(x_l==x_u)
		#list_changed_var=np.unique(list_changed_var)
		if len(list_changed_var)==0:
			break
		
		list_constraints_to_check2=[]
		for i in list_changed_var:
			#to_add=np.nonzero(Acsc[:,i])[0]
			to_add2=Acsc.indices[Acsc.indptr[i]:Acsc.indptr[i+1]]
			#assert(np.all(to_add==to_add2))
			list_constraints_to_check2.append(to_add2)
		list_constraints_to_check2=np.unique(np.hstack(list_constraints_to_check2))
		list_changed_var=[]
		#list_constraints_to_check=np.arange(Acsr.shape[0])
		for j in list_constraints_to_check2:
			#line=Acsr[j,:]# very slow...
			#indices=line.indices
			#data=line.data
			indices=Acsr.indices[Acsr.indptr[j]:Acsr.indptr[j+1]]
			data=Acsr.data[Acsr.indptr[j]:Acsr.indptr[j+1]]

			
			interval_l=0
			interval_u=0
			for k in range(indices.size):
				i=indices[k]
				v=data[k]
				if v>0:
					interval_u+=v*x_u[i]
					interval_l+=v*x_l[i]
				else: 
					interval_l+=v*x_u[i]
					interval_u+=v*x_l[i]				
					
			if interval_u<b_lower[j] or interval_l>b_upper[j]:
				return 0,j
					
			for k in range(indices.size):
				i=indices[k]
				v=data[k]			
				if v>0:
					
					n_u=np.floor(tol+(b_upper[j]-interval_l+v*x_l[i])/v)
					n_l=np.ceil(-tol+(b_lower[j]-interval_u+v*x_u[i])/v)
				else: 
					n_u=np.floor(tol+(b_lower[j]-interval_u+v*x_l[i])/v)
					n_l=np.ceil(-tol+(b_upper[j]-interval_l+v*x_u[i])/v)
									
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
					list_changed_var.append(i)
					#assert(j in list_constraints_to_check2)
					
					
					
	#print '%d variable fixed '% np.sum(x_l==x_u)
	return 1,None



	
	

def revert(back_ops,x_l,x_u):
	for t,i,v in reversed(back_ops):
		if t==0:
			x_l[i]=v
		else:
			x_u[i]=v	 
	
	
#@profile	
def greedy_round(x,LP,callbackFunc=None,maxiter=np.inf,order=None,fixed=None):
	#save_arguments('greedy_round_test')
	if False:
		import pickle
		d={'x':x,'LP':LP}
		with open('greedy_test.pkl', 'wb') as f:				
			pickle.dump(d,f)	
	if not callbackFunc is None:
		callbackFunc(0,np.round(x),0,0,0,0,0)
	LP2=copy.copy(LP)
	LP2.convertToAllInequalities()
	assert(LP2.Aequalities is None)
	
	
	x_u=LP2.upperbounds.copy()
	x_l=LP2.lowerbounds.copy()
	
	if not fixed is None:
		x_l[fixed]=x[fixed]
		x_u[fixed]=x[fixed]
	

	A=LP2.Ainequalities 
	b_l=LP2.B_lower.copy()
	b_u=LP2.B_upper.copy()
		
	#callbackFunc(0,np.maximum(x_r.astype(np.float),0),0,0,0,0,0)
	A_csr=A.tocsr()	
	A_csc=A.tocsc()
	if order is None:
		# sort from the less fractional to the most fractional
		# order=np.argsort(np.abs(x-np.round(x))+c*np.round(x))
		order=np.argsort(LP2.costsvector*(2*np.round(x)-1))
		#order=np.argsort(LP2.costsvector*np.round(x))
		#order=np.arange(x.size)
		#order=np.arange(x.size)[::-1]
	#x_r=np.full(x.size,-1,dtype=np.int32)
	x_r=x.copy()
	mask=np.zeros(x.size,dtype=np.int32)
	depth=0
	nb_backtrack=0
	#callbackFunc(0,x,0,0,0,0,0)
	
	
	valid,idcons=propagateConstraints(np.arange(A.shape[1]),x_l, x_u, A_csr, A_csc, b_l, b_u,[])
	if valid==0:
		return x_r,valid
		

	#check that no constraint is violated
	back_ops=[[] for i in range(x.size)]
	niter=0
	while depth<x.size :
		niter+=1
		#callbackFunc(0,x_l,0,0,0,0,0)
		#print depth
		
		idvar=order[depth]
		#print mask[order]
		if mask[idvar]==2:
			mask[idvar]=0
			revert(back_ops[depth],x_l,x_u)
			depth=depth-1
			revert(back_ops[depth],x_l,x_u)
			print 'step back to depth %d'%depth
			continue
		
		if x_u[idvar]==x_l[idvar]:# the variable is already fixed thanks to constraint propagation
			back_ops[depth]=[]
			depth=depth+1
			x_r[idvar]=x_u[idvar]
			mask[idvar]=2
		elif mask[idvar]==0:
			x_r[idvar]=np.round(x[idvar])
			mask[idvar]=1
			back_ops[depth]=[]
			back_ops[depth].append((1,idvar,x_u[idvar]))
			back_ops[depth].append((0,idvar,x_l[idvar]))
			x_u[idvar]=x_r[idvar]
			x_l[idvar]=x_r[idvar]
			
			#violated_eq=check_constraints(idvar,x_r,mask,Aeq_csr,Aeq_csc,beq,beq)
			#violated_ineq=check_constraints(idvar,x_r,mask,Aineq_csr,Aineq_csc,b_lower,b_upper)
			#violated=violated_eq | violated_ineq
			valid,idcons=propagateConstraints([idvar],x_l, x_u, A_csr, A_csc, b_l, b_u,back_ops[depth])
			
			x_l[idvar]
			if valid:
				#valid,back_ops_init2=propagateConstraints(np.arange(A.shape[1]),x_l, x_u, A_csr, A_csc, b_l, b_u,[])
				#assert(len(back_ops_init2)==0)
				depth=depth+1
				
			else:	
				revert(back_ops[depth],x_l,x_u)
					
		elif mask[idvar]==1:
				
			x_r[idvar]=1-round(x[idvar])
			back_ops[depth]=[]
			back_ops[depth].append((1,idvar,x_u[idvar]))
			back_ops[depth].append((0,idvar,x_l[idvar]))
			x_u[idvar]=x_r[idvar]
			x_l[idvar]=x_r[idvar]				
			
			mask[idvar]=2
			valid,idcons=propagateConstraints([idvar],x_l, x_u, A_csr, A_csc, b_l, b_u,back_ops[depth])
			if valid:
				#valid,back_ops_init2=propagateConstraints(np.arange(A.shape[1]),x_l, x_u, A_csr, A_csc, b_l, b_u,[])
				#assert(len(back_ops_init2)==0)			
				depth=depth+1	
				
			else:
				mask[idvar]=0
				#callbackFunc(0,x_l,0,0,0,0,0)
				#name=LP.getInequalityConstraintNameFromId(idcons)['name']				
				#print 'constaint %d of type %s violated,steping back to depth %d'%(idcons,name,depth)
				
				x_l2=x_l.copy()*0.5
				x_l2[idvar]=1
				#callbackFunc(0,x_l2,0,0,0,0,0)
				
				revert(back_ops[depth],x_l,x_u)
				depth=depth-1
				nb_backtrack+=1
				revert(back_ops[depth],x_l,x_u)
				
				
				#raise # need a way to save the bound constraint to restore it

				#raise # need a way to save the bound constraint to restore it
	#callbackFunc(0,np.maximum(x_r.astype(np.float),0),0,0,0,0,0)
	valid=propagateConstraints(np.arange(A.shape[1]),x_l, x_u, A_csr, A_csc, b_l, b_u,[])
	#assert(valid)
	print 'backtracked %d times'%nb_backtrack
	print 'energy after rounding =%f'%np.sum(x_r*LP.costsvector)
	return x_r,valid




	
	
	
def greedy_fix(x,LP,nbmaxiter=1000,callbackFunc=None,useXorMoves=False):
	# decrease the constraints violation score using coordinate descent
	
	xr=np.round(x)
	
	LP2=copy.copy(LP)
	
	#xors=np.nonzero(LP.B_lower==1)[0]
	
	#xors=np.nonzero(LP.Bequalities==1)[0]
	#assert(np.all(LP.Aequalities[xors,:].data==1))
	
	
	LP2.convertToAllInequalities()	
	LP2.convertToOnesideInequalitySystem()
	
	
	
	assert np.all(xr<=LP2.upperbounds)
	assert np.all(xr>=LP2.lowerbounds)
	
	assert(LP2.B_lower is None)
	# compute the sum of the of violated constraints with the magnitude of the violation
	AinequalitiesCSC=LP2.Ainequalities.tocsc()
	constraints_costs=np.ones(AinequalitiesCSC.shape[0])
	#constraints_costs[:]=0.2
	xors=LP2.findInequalityConstraintsFromName('xors')
	for item in xors:
		constraints_costs[item['start']:item['end']+1]=1000
	#for item in xors:
	# print np.max(r_ineq_threholded[item['start']:item['end']+1])
	r_ineq = LP2.Ainequalities*xr-LP2.B_upper
	r_ineq_threholded=np.maximum(r_ineq,0)
	score_ineq=np.sum(r_ineq_threholded*constraints_costs)
	# test switching single variable
	#constraints_gradient=r_ineq_theholded*LP2.Ainequalities
	
	score_decrease=np.zeros(x.size)
	
	
	R=LP2.Ainequalities.copy()
	R.data=np.random.rand(R.data.size)
	tocheck=np.nonzero(r_ineq_threholded*R!=0)[0]
	check=False
	
	Dx=scipy.sparse.csc.csc_matrix((1-2*xr,(np.arange(xr.size),np.arange(xr.size))),(xr.size,xr.size))
	dr_ineq_matrix=AinequalitiesCSC*Dx
	
	if useXorMoves:
		
		#adding xor moves
		#XorDx=
		xormoves=[]
		xorid_to_moves_interval=np.zeros(LP2.Ainequalities.shape[0])
		for xorsintervals in xors:
			for r in range(xorsintervals['start'],xorsintervals['end']+1):
				ids=LP2.Ainequalities[r,:].indices
				data=LP2.Ainequalities[r,:].data
				assert(len(ids)==4)
				
				vec=-xr[ids]
				xorid_to_moves_interval[r]=len(xormoves)
				for i,id in enumerate(ids):				
					vec2=vec.copy()
					vec2[i]+=1				
					xormoves.append((ids,vec2,r))
					
		xor_score_decrease=np.zeros(len(xormoves))			
		for i,move in enumerate(xormoves):
			for j,id in enumerate(move[0]):
				new_r_ineq=r_ineq[id]+move[1][j]
				new_r_ineq_threholded=np.maximum(new_r_ineq,0)
				xor_score_decrease[i]+=(new_r_ineq_threholded-r_ineq_threholded[id])*constraints_costs[id]
							
				
			
		
		
	
	for iter in range(nbmaxiter):
		if check:
			r_ineq = LP2.Ainequalities*xr-LP2.B_upper
			r_ineq_threholded=np.maximum(r_ineq,0)
			score_ineq=np.sum(r_ineq_threholded*constraints_costs)	
		
		
		dr_ineq_matrix=AinequalitiesCSC*Dx[:,tocheck]
		
		for j,i in enumerate(tocheck):
			
			#dxi=1-2*xr[i]
			#dr_ineq=AinequalitiesCSC[:,i]*dxi
			#dx=scipy.sparse.csc.csc_matrix(([1-2*xr[i]],([i],[0])),(xr.size,1))
			#dr_ineq=AinequalitiesCSC*dx
			score_decrease[i]=0
			dr_ineq=dr_ineq_matrix[:,j]
			assert(dr_ineq.format=='csc')
			
			for j,id in enumerate(dr_ineq.indices):
				new_r_ineq=r_ineq[id]+dr_ineq.data[j]
				new_r_ineq_threholded=np.maximum(new_r_ineq,0)
				score_decrease[i]+=(new_r_ineq_threholded-r_ineq_threholded[id])*constraints_costs[id]
				
			
			
			if check:
			
				xr2=xr.copy()
				xr2[i]=1-xr2[i]
				r_ineq2 = LP2.Ainequalities*xr2-LP2.B_upper
				
				#r_ineq2b=dr_ineq.toarray().flatten()+r_ineq
				#np.max(np.abs(new_r_ineq_threholded-r_ineq_threholded2))
				#np.max(np.abs(new_r_ineq-r_ineq2))
				#np.max(np.abs(r_ineq2b-r_ineq2))
				r_ineq_threholded2=np.maximum(r_ineq2,0)
				score_ineq2=np.sum(r_ineq_threholded2*constraints_costs)	
				score_decrease2=score_ineq2-score_ineq
				assert(score_decrease2==score_decrease[i])
				
			#if score_decrease[i]<0:
				#print "found move"
					
		
		#swith the variable that decreases the most the constraints violation score
		# 
		if min(score_decrease)>=0:
			print 'could not find more moves'
			if not callbackFunc is None:
				callbackFunc(0,xr,0,0,0,0,0)			
			
			r_ineq = LP2.Ainequalities*xr-LP2.B_upper
			r_ineq_threholded=np.maximum(r_ineq,0)
			tocheck=np.nonzero(r_ineq_threholded*R!=0)[0]
			tocheck=np.arange(xr.size)
			score_ineq2=np.sum(r_ineq_threholded*constraints_costs)	
			assert(score_ineq2==score_ineq)
			return xr
		
		
		
		ibest=np.argmin(score_decrease)
		#idbestxormove=np.argmin(xor_score_decrease)
		# 
		
		
		
		#r_ineq=
		i=ibest
		#dxi=1-2*xr[i]
		
		#dr_ineq=AinequalitiesCSC[:,i]*dxi
		dr_ineq=AinequalitiesCSC*Dx[:,i]
		
		score_decrease_best=0
		for j,id in enumerate(dr_ineq.indices):
			r_ineq[id]=r_ineq[id]+dr_ineq.data[j]
			new_r_ineq_threholded=np.maximum(r_ineq[id],0)
			score_decrease_best+=(new_r_ineq_threholded-r_ineq_threholded[id])*constraints_costs[id]
			r_ineq_threholded[id]=new_r_ineq_threholded
			
		assert(np.abs(score_decrease_best-score_decrease[ibest])<1e-8)
		
		score_ineq+=score_decrease_best
		print score_ineq
		#xr[ibest]=1-xr[ibest]
		dx=Dx[:,ibest]
		xr[dx.indices]+=dx.data
		if not callbackFunc is None:
			callbackFunc(0,xr,0,0,0,0,0)
		
		# update switching score of variables that may have changed	
		#tocheck=np.nonzero(dr_ineq.T*R!=0)[1]
		movetochange=(dx.T*Dx).indices
		Dx[:,movetochange]=scipy.sparse.csc.csc_matrix((1-2*xr[movetochange],(movetochange,np.arange(movetochange.size))),(xr.size,movetochange.size))
		tocheck=np.nonzero(dr_ineq.T*R*Dx!=0)[1]
		
	
	

		
