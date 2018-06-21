import numpy as np
# For reference, please refer to "On the Implementation of a Primal-Dual 
# Interior Point Method" by Sanjay Mehrotra.
# https://www.researchgate.net/publication/230873223_On_the_Implementation_of_a_Primal-Dual_Interior_Point_Method
# this code is largely inspired from https://github.com/YimingYAN/mpc

from scipy.sparse.linalg import norm as snorm
from scipy import sparse
from numpy.linalg import norm
#from scikits.sparse.cholmod import cholesky
#import scikits.sparse.cholmod 
from pysparselp.xorshift import xorshift

def initialPoint(A,b,c):
    
    n = A.shape[1]
    e = np.ones((n,))
    
    # solution for min norm(s) s.t. A'*y + s = c
    #y =sparse.linalg.cg(A*A.T, A*c,tol=1e-7)[0]
    y =sparse.linalg.spsolve(A*A.T, A*c)
    
    #y2 =sparse.linalg.cgs(A*A.T, A*c)[0]
    #y2 =sparse.linalg.gmres(A*A.T, A*c,)[0]
    
    s = c-A.T*y
    
    # solution for min norm(x) s.t. Ax = b 
    x = A.T*sparse.linalg.spsolve(A*A.T, b)
    #x = A.T*sparse.linalg.cg(A*A.T, b,tol=1e-7)[0]
    
    # delta_x and delta_s
    delta_x = max(-1.5*np.min(x),0)
    delta_s = max(-1.5*np.min(s),0)
    
    # delta_x_c and delta_s_c
    pdct = 0.5*(x+delta_x*e).dot(s+delta_s*e)
    delta_x_c = delta_x+pdct/(np.sum(s)+n*delta_s)
    delta_s_c = delta_s+pdct/(np.sum(x)+n*delta_x)
    
    # output
    x0 = x+delta_x_c*e
    s0 = s+delta_s_c*e
    y0 = y
    return x0, y0, s0
    
    
    
def newtonDirection(Rb, Rc, Rxs, A, m, n, x, s, lu, errorCheck=0):
    
    rhs =np.hstack((-Rb,-Rc+Rxs/x))
    D_2 = -np.minimum(1e+16, s/x)
    B = sparse.vstack ((sparse.hstack((sparse.coo_matrix((m,m)), A)), sparse.hstack((A.T, sparse.diags([D_2], [0])))))
    
    # ldl' factorization
    # if L and D are not provided, we calc new factorization; otherwise,
    # reuse them
    useLu=True
    if useLu:
        if (lu is None)  :
            lu = sparse.linalg.splu(B.tocsc())
            # wikipedia says it uses Mehrotra cholesky but the matrix i'm getting is not definite positive
            # scikits.sparse.cholmod.cholesky fails without a warning 
    
        sol=lu.solve(rhs)
    else:
         sol=sparse.linalg.cg(B,rhs,tol=1e-5)[0]
         #assert(np.max(np.abs(B*sol-rhs))<1e-5)
      
        
        
    dy = sol[:m]
    dx = sol[m:m+n];
    ds = -(Rxs+s*dx)/x;
    
    if errorCheck == 1:
        print ('error = %6.2e'%(norm(A.T*dy + ds + Rc)+ norm(A*dx + Rb) + norm(s*dx + x*ds + Rxs)),)
        print ('\t + err_d = %6.2e'%(norm(A.T*dy + ds + Rc)),)
        print ('\t + err_p = %6.2e'%(norm(A*dx + Rb)),)
        print ('\t + err_gap = %6.2e\n'%(norm(s*dx + x*ds + Rxs)),)
      
    return dx, dy, ds, lu
    
def  stepSize(x, s, Dx, Ds, eta= 0.9995):
    alphax = -1/min(min(Dx/x),-1) 
    alphax = min(1, eta * alphax)
    alphas = -1/min(min(Ds/s),-1)
    alphas = min(1, eta * alphas)
    return alphax, alphas 




def   mpcSol(A, b, c, maxN=100,eps= 1e-08,theta=0.9995,verbose=2,errorCheck=False,callBack=None):
    
    A=sparse.coo_matrix(A)
    c=np.squeeze(np.array(c))
    b=np.squeeze(np.array(b))
  
    
    # Initialization
    
    m,n = A.shape
    alphax = 0
    alphas = 0
    
  
    if verbose > 1:
        print('\n%3s %6s %11s %9s %9s\n'%('ITER', 'MU', 'RESIDUAL', 'ALPHAX', 'ALPHAS'))
 
    # Choose initial point
    x, y, s = initialPoint(A, b, c)
    
    bc = 1+max([norm(b),norm(c)])
    
    # Start the loop
    for iter in range(maxN):
        # Compute residuals and update mu
        Rb = A*x-b
        Rc = A.T*y+s-c
        Rxs = x*s
        mu = np.mean(Rxs)
        
        # Check relative decrease in residual, for purposes of convergence test
        residual  = norm(np.hstack((Rb,Rc,Rxs))/bc)
        
        if verbose > 1:
            print ('%3d %9.2e %9.2e %9.4g %9.4g'%(iter, mu, residual, alphax, alphas))
            
        if not callBack is None:
            callBack(x,iter)
        
        if residual < eps:
            break
       
        
        # ----- Predictor step -----
        
        # Get affine-scaling direction
        dx_aff, dy_aff, ds_aff, lu = newtonDirection(Rb, Rc, Rxs, A, m, n,x, s,  None,errorCheck)
        
        # Get affine-scaling step length
        alphax_aff, alphas_aff = stepSize(x, s, dx_aff, ds_aff, 1)
        mu_aff = (x+alphax_aff*dx_aff).dot(s+alphas_aff*ds_aff)/n
        
        # Set central parameter
        sigma = (mu_aff/mu)**3;
        
        # ----- Corrector step -----
        
        # Set up right hand sides
        Rxs = Rxs + dx_aff*ds_aff - sigma*mu*np.ones((n))
        
        # Get corrector's direction
        dx_cc, dy_cc, ds_cc, lu =newtonDirection(Rb, Rc, Rxs, A, m, n, x, s, lu,errorCheck)
        
        # Compute search direction and step
        dx = dx_aff+dx_cc
        dy = dy_aff+dy_cc
        ds = ds_aff+ds_cc
        
        alphax, alphas =  stepSize(x, s, dx, ds,theta)
        
        # Update iterates
        x = x + alphax*dx
        y = y + alphas*dy
        s = s + alphas*ds
        
        if iter == maxN and param.verbose > 1:
            print ('maxN reached!\n');
    
    if verbose > 0:
        print ('\nDONE! [m,n] = [%d, %d], N = %d\n'%(m,n,iter))
    
    N = iter
    f = c.T.dot(x)
    return f, x, y, s, N



if __name__ == "__main__":
   
    m = 100
    n = 120
  
    r=xorshift()
    A = np.matrix(r.randn(m,n))
    b = A*r.rand(n,1)
    c = A.T*r.rand(m,1) 
    c=c+r.rand(n,1)
    f, x, y, s, N = mpcSol(A, b, c)   