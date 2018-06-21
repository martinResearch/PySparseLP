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



import copy
import numpy as np
import time
import scipy.sparse
import scipy.ndimage

#@profile
def ChambollePockPPDAS(LP,x0=None,\
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
	if not x0 is None:
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
		if not Aeq is None:
			print ("Aeq shape="+str(Aeq.shape))
	
			assert(scipy.sparse.issparse(Aeq))  
			assert(Aeq.shape[1]==c.size)
			assert(Aeq.shape[0]==beq.size)
			AeqCopy=Aeq.copy()               
			AeqCopy.data=np.abs(AeqCopy.data)**(2-alpha)        
			SumAeq=(np.ones((1,AeqCopy.shape[0]))*AeqCopy)  
			tmp=tmp+SumAeq
			#AeqT=Aeq.T
		if not Aineq is None:     
			print( "Aineq shape="+str(Aineq.shape))
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
		
		
	if not Aeq is None:
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
	if not Aineq is None:  
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
		
		if not Aeq is None:				
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
				

		if not Aineq is None: 			
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
			if not Aeq is  None:
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
			if not Aeq is None:
				r_eq=(AeqCSC*x3)-beq
			r_ineq=(AineqCSC*x3)-bineq	
			#tmp=np.abs(diagSigma_ineq*r_ineq)*((y_ineq>0) | (diagSigma_ineq*r_ineq>0 ))
			#list_active_inequality_constraints,=np.nonzero(tmp>np.percentile(tmp, 10))
			is_active_inequality_constraint= (r_ineq>-0.2) | (y_ineq>0)
			list_active_inequality_constraints,=np.nonzero(is_active_inequality_constraint)
			
			if not Aeq is None:
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
			if not Aeq is None:
				r_eq_active=r_eq[list_active_equality_constraints]
			#subAineqCSC=AineqCSC[list_active_inequality_constraints,:]
			#subAineqCSC=AineqCSR[list_active_inequality_constraints,:].tocsc()
			if not Aeq is None:
				subAeqCSR=AeqCSR[list_active_equality_constraints,:]
				subAeqCSC2=subAeqCSR.tocsc()[:,list_active_variables]
				subAeqCSR2=subAeqCSC2.tocsr()
			subAineqCSR=AineqCSR[list_active_inequality_constraints,:]
			
			
			subAineqCSC2=subAineqCSR.tocsc()[:,list_active_variables]
			
			subAineqCSR2=subAineqCSC2.tocsr()
			r_ineq_active=r_ineq[list_active_inequality_constraints]
			if i%nb_iter_plot==0:
				print ('%d active variables %d  active inequalities %d active equalities'%(np.sum(is_active_variable),np.sum(is_active_inequality_constraint),nb_active_equality_constraints))
			
				print ('%f percent of active variables %f percent active inequalities %f percent active equalities'%(100*np.mean(is_active_variable),100*np.mean(is_active_inequality_constraint),percent_active_equality_constraint))
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
		if not Aeq is None:
			if use_vec_sparsity:
				r_eq=(Aeq*x3_sparse).toarray().ravel()-beq
			else:	
				
				#r_eq=r_eq+(Aeq[:,list_active_variables]*diff_active_x)# can use sparisity in diff_x3
				#r_eq=r_eq+Aeq*sparse_diff_x
				#increment=subAeqCSC*sparse_diff_x # to do : update only the active constraints residuals
				#r_eq_active[increment.indices]=r_eq_active[increment.indices]+increment.data 
				
				
				r_eq_active+=subAeqCSC2*diff_active_x3
				#r_eq=r_eq+increment.toarray().ravel()
						
		if not Aineq is None: 			
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
			if not Aeq is None:
				r_eq=(AeqCSC*x)-beq
			
			
			prev_elapsed=elapsed
			elapsed= (time.clock() - start)	
			mean_iter_priod=(elapsed-prev_elapsed)/10
			if (not max_time is None) and elapsed>max_time:
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
			if not Aeq is None:
				energy1+=y_eq.T.dot(Aeq*x-beq)
				energy2-=y_eq.dot(beq)
				max_violated_equality=np.max(np.abs(r_eq))
			if not Aineq is None:    
				energy1+=y_ineq.T.dot(Aineq*x-bineq)
				energy2-=y_ineq.dot(bineq)
				max_violated_inequality=np.max(r_ineq)
				
		

			xrounded=np.round(x)
			#xrounded=greedy_round(x,c,Aeq,beq,Aineq,np.full(bineq.shape,-np.inf),bineq,lb.copy(),ub.copy(),callbackFunc=callbackFunc)
			
			energy_rounded=c.dot(xrounded)
			if not Aeq is None:
				nb_violated_equality_rounded=np.sum(np.abs(Aeq*xrounded-beq))
			else:
				nb_violated_equality_rounded=0
			if not Aineq is None:	
				nb_violated_inequality_rounded=np.sum(np.maximum(Aineq*xrounded-bineq,0))	
			else:
				nb_violated_inequality_rounded=0
			
			if nb_violated_equality_rounded==0 and nb_violated_inequality_rounded==0:
				print ('##########   found feasible solution with energy'+str(energy_rounded))
				if energy_rounded<best_integer_solution_energy:
					best_integer_solution_energy=energy_rounded
					best_integer_solution=xrounded
				

			print ('iter'+str(i)+": energy1= "+str(energy1) + " energy2="+str(energy2)+ ' elaspsed '+str(elapsed)+' second'+\
				  ' max violated inequality:'+str(max_violated_inequality)+\
				  ' max violated equality:'+str(max_violated_equality)+\
				  'mean_iter_period='+str(mean_iter_priod)+\
			'rounded : %f ineq %f eq'%(nb_violated_inequality_rounded,nb_violated_equality_rounded))
				#'y_eq has '+str(100 * np.mean(y_eq==0))+' % of zeros '+\
		#    'y_ineq has '+str(100 * np.mean(y_ineq==0))+' % of zeros '+\			 
		
			if not callbackFunc is None:

				callbackFunc(i,x,energy1,energy2,elapsed,max_violated_equality,max_violated_inequality,is_active_variable=is_active_variable)

		#Update the dual variables
	
		if not Aeq is None:
			diff_active_y_eq=diagSigma_eq[list_active_equality_constraints]*r_eq_active
			y_eq[list_active_equality_constraints]=y_eq[list_active_equality_constraints]+diff_active_y_eq
			
			#y_eq=y_eq+diagSigma_eq*r_eq
			#y_eq+=diagSigma_eq*r_eq

		if not Aineq is None: 
			#active_y_ineq=y_ineq[list_active_inequality_constraints]
			new_active_y_ineq=active_y_ineq+diagSigma_ineq_active*r_ineq_active
			new_active_y_ineq=np.maximum(new_active_y_ineq, 0)			
			diff_active_y_ineq=new_active_y_ineq-active_y_ineq
			active_y_ineq=new_active_y_ineq# np.mean(diff_active_y_ineq!=0) often give me 0.05 on the facade , can i use that for more speedups ?
			#y_ineq[list_active_inequality_constraints]=active_y_ineq
			
			#y_ineq+=diagSigma_ineq*r_ineq			
			#np.maximum(y_ineq, 0,y_ineq) 
	
			
	
		
	return x[:n]

