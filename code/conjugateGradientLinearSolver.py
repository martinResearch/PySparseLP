def conjgrad(A,b,x0,maxiter=100,tol=1e-10):
""" This function solves Ax=b using the conjugate gradient method
    converted from the matlab code from wikipedia
    wikipedia http://en.wikipedia.org/wiki/Conjugate_gradient_method
"""
	x=x0.copy()
	r=b-A*x
	p=r
	rsold=r.dot(r)
	
	for i in range(maxiter):
		Ap=A*p
		alpha=rsold/(p.dot(Ap))
		x=x+alpha*p
		r=r-alpha*Ap
		rsnew=r.dot(r)
		if np.sqrt(rsnew)<tol:
			break
			
		p=r+(rsnew/rsold)*p;
		rsold=rsnew
	return x 
