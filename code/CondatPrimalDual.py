			
	
	


	


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
