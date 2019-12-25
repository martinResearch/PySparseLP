"""
Sparse inverse covariance estimation
"""



import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf
import matplotlib.pyplot as plt
from pysparselp.SparseLP import SparseLP

class SparseInvCov(SparseLP):



    def addAbsPenalization(self,I,coefpenalization):
       
        aux=self.addVariablesArray(I.shape,upperbounds=None,lowerbounds=0,costs=coefpenalization)
       
        if np.isscalar(coefpenalization):
            assert(coefpenalization>0)			
        else:#allows a penalization that is different for each edge (could be dependent on an edge detector)
            assert(coefpenalization.shape==aux.shape)
            assert(np.min(coefpenalization)>=0)			
        aux_ravel=aux.ravel()
        I_ravel=I.ravel()
        cols=np.column_stack((I_ravel,aux_ravel))
        vals=np.tile(np.array([1,-1]),[I.size,1])	
        self.addLinearConstraintRows(cols,vals,lowerbounds=None,upperbounds=0)			
        vals=np.tile(np.array([-1,-1]),[I.size,1])  
        self.addLinearConstraintRows(cols,vals,lowerbounds=None,upperbounds=0) 
        
        
def run():
    
    ##############################################################################
    # Generate the data
    n_samples = 40
    n_features = 20
    
    prng = np.random.RandomState(1)
    prec = make_sparse_spd_matrix(n_features, alpha=.98,
                                  smallest_coef=.4,
                                  largest_coef=.7,
                                  random_state=prng)
    cov = linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    
    ##############################################################################
    # Estimate the covariance
    emp_cov = np.dot(X.T, X) / n_samples
    
    LP=SparseInvCov()
    ids=LP.addVariablesArray(shape=emp_cov.shape, lowerbounds=None, upperbounds=None)
    lamb=0.15
    from scipy import sparse
    
    C=sparse.kron(sparse.csr_matrix(emp_cov), sparse.eye(n_features))
    LP.addConstraintsSparse(C,np.eye(emp_cov.shape[0]).flatten()-lamb,np.eye(emp_cov.shape[0]).flatten()+lamb)
    LP.addAbsPenalization(ids,1)
    x=LP.solve(method='Mehrotra',nb_iter=60000,max_time=20)[0]
    #x=LP.solve(method='ChambollePockPPD')[0]
    lp_prec_=x[ids]
    lp_prec_=0.5*(lp_prec_+lp_prec_.T)
    plt.figure()
    vmax = .9 * prec.max()
    lp_prec_=lp_prec_*(np.abs(lp_prec_)>1e-8)
    this_prec=lp_prec_
    lp_cov_=np.linalg.inv(lp_prec_)
    
    
    
    
    ##############################################################################
    # Plot the results
    plt.ion()
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)
    
    # plot the covariances
    covs = [('Empirical', emp_cov), ('LP', lp_cov_),
            ('True', cov)]
    vmax = cov.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.subplot(2, 3, i + 1)
        plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s covariance' % name)
    
    
    # plot the precisions
    precs = [('Empirical', linalg.inv(emp_cov)), ('LP', lp_prec_),
              ('True', prec)]
    vmax = .9 * prec.max()
    for i, (name, this_prec) in enumerate(precs):
        ax = plt.subplot(2, 3, i + 4)
        plt.imshow(np.ma.masked_equal(this_prec, 0),
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s precision' % name)
        ax.set_facecolor ('.7')
        
    plt.tight_layout()
    
    
    plt.show()
    
if __name__ == "__main__":
    run()

    
