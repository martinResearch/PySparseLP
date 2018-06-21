from SparseLP import SparseLP,solving_methods
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import copy
import time
from sklearn import datasets, linear_model,ensemble
from sklearn.neural_network import MLPRegressor


def randSparse(shape,sparsity):
    if isinstance(shape,tuple) or isinstance(shape,list):
        return np.round(np.random.randn(*shape)*100)*(np.random.rand(*shape)<sparsity)/100
    else:
        return  np.round(np.random.randn(shape)*100)*(np.random.rand(shape)<sparsity)/100

def generateRandomLP(nbvar,n_eq,n_ineq,sparsity):
    
    # maybe could have a look at https://www.jstor.org/stable/3689906?seq=1#page_scan_tab_contents
    # https://deepblue.lib.umich.edu/bitstream/handle/2027.42/3549/bam8969.0001.001.pdf
    feasibleX=randSparse(nbvar,sparsity=1)
    
    if n_ineq>0:
        while True : # make sure the mattrix is not empy=ty
            A_ineq=scipy.sparse.csr_matrix(randSparse((n_ineq,nbvar),sparsity))
            keep=((A_ineq!=0).dot(np.ones(nbvar)))>=2 # keep only rows with at least two non zeros values 
            if np.sum(keep)>=1:
                break
        bmin=A_ineq.dot(feasibleX)
        b_upper=np.ceil((bmin+abs(randSparse(n_ineq,sparsity)))*1000)/1000# make v feasible
        b_lower=None#bmin-abs(randSparse(n_ineq,sparsity))
        A_ineq=A_ineq[keep,:]
        b_upper=b_upper[keep]    
       
        
    costs=randSparse(nbvar,sparsity=1)

    t=randSparse(nbvar, sparsity=1)
    lowerbounds=feasibleX+np.minimum(0, t)
    upperbounds=feasibleX+np.maximum(0, t)
    
    LP=SparseLP()
    LP.addVariablesArray(nbvar,lowerbounds=lowerbounds,
                         upperbounds=upperbounds,
                         costs=costs)
    if n_eq>0:
        Aeq=scipy.sparse.csr_matrix(randSparse((n_eq,nbvar),sparsity))
        Beq=Aeq.dot(feasibleX)   
        keep=((Aeq!=0).dot(np.ones(nbvar)))>=2 # keep only rows with at least two non zeros values
        Aeq=Aeq[keep,:]
        Beq=Beq[keep]
        if Aeq.indices.size>0:
            LP.addEqualityConstraintsSparse(Aeq,Beq)
    if  n_ineq>0 and A_ineq.indices.size>0:
        LP.addConstraintsSparse(A_ineq,b_lower,b_upper) 
    
    assert(LP.checkSolution (feasibleX) )      
    return LP,feasibleX



if __name__ == "__main__":
    plt.ion()
   
    LP,v=generateRandomLP(nbvar=30,n_eq=1,n_ineq=30,sparsity=0.2)
    LP2=copy.deepcopy(LP)
    LP2.convertToOnesideInequalitySystem()
    scipySol,elapsed=LP2.solve(method='ScipyLinProg',force_integer=False,getTiming=True,nb_iter=100000)
    costScipy=scipySol.dot(LP2.costsvector.T)
    maxv=LP2.maxConstraintViolation(scipySol)
    if maxv>1e-8:
        print ('not expected')
        raise
    
    groundTruth=scipySol
    solving_methods2=list(set(solving_methods) - set(['DualGradientAscent']))
   
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].set_title('mean absolute distance to solution')
    axarr[1].set_title('maximum constraint violation')
    axarr[2].set_title('difference with optimum value')  
    max_time=2
    for i,method in enumerate(solving_methods2):
        sol1,elapsed=LP2.solve(method=method,max_time=max_time,groundTruth=groundTruth)
        axarr[0].semilogy(LP2.opttime_curve,np.maximum(LP2.distanceToGroundTruth,1e-18),label=method)
        axarr[1].semilogy(LP2.opttime_curve,np.maximum(LP2.max_violated_constraint,1e-18)) 
        axarr[2].semilogy(LP2.opttime_curve,np.maximum(LP2.pobj_curve-costScipy,1e-18))
        axarr[0].legend()
        plt.show()
    print ('done' )  
    
    
    
    
    
    
    
    