# Goal

This project provides several python codes to solve very sparse linear programs of the form

![latex:\large $\mathbf{x}^*=argmin_\mathbf{x} \mathbf{c}^t\mathbf{x} ~  s.t.~  A_e\mathbf{x}=\mathbf{b_e},A_i\mathbf{x}\leq\mathbf{ b_i}, \mathbf{l}\leq \mathbf{x}\leq \mathbf{u}$ ](https://rawgithub.com/martinResearch/PySparseLP/master/images/LPproblem.svg)

The different algorithms that are implemented are documented in the [pdf](./latex/SparseLinearProgramming.pdf): 

* a dual coordinate ascent method with exact line search 
* a dual gradient ascent with exact line search
* a first order primal-dual algorithm adapted from chambolle pock [2]
* three methods based on the Alternating Direction Method of Multipliers [3]

**Note** These methods are not meant to be efficient methods to solve generic linear programs. They are simple and quite naive methods i coded while exploring different possibilities to solve very large sparse linear programs that are too big to be solved using the standard simplex method or standard interior point methods.


This project also provides: 

* a python implementation of Mehrotra's Predictor-Corrector Pimal-Dual Interior Point method.
* a python class *SparseLP* (in SparseLP.py) that  makes it easier to build linear programs from python 
* methods to convert between the different common forms of linear programs (slack form , standard form etc), 
* methods to import and export the linear program from and to standard file formats (MPS), It is used here to run [netlib](http://www.netlib.org/lp/data/) LP problems.
* a simple constraint propagation method with back-tracking to find feasible integer values solutions (for integer programs)
* interfaces to other solvers (SCS, ECOS, CVXOPT) through CVXPY

# Installation


using pip

	sudo pip install git+git://github.com/martinResearch/PySparseLP.git
	
otherwise you can dowload it, decompressit it  and compile it locally using 

	python setup.py build_ext --inplace

# Examples

## Image segmentation
we can use it to solve a binary image segmentation problem with Potts regularisation.

![latex: \large $min_s c^ts + \sum_{(i,j)\in E}  |s_i-s_j| ~s.t. ~0 \leq s\leq 1$](https://rawgithub.com/martinResearch/PySparseLP/master/images/segmentation.svg)

with *E* the list of indices of pairs of neighbouring pixels and *c* a cost vector that is obtain from color distribution models of the two regions.
This problem can be rewritten as a linear progam by adding an auxiliary variable *d_ij* for each edge with the constraints

![latex: \large $min_s c^ts + \sum_{(i,j)\in E}  d_{ij} ~s.t. ~0 \leq s\leq 1, ~d_{ij}\geq s_j-s_j, ~d_{ij}\geq s_i-s_i $](https://rawgithub.com/martinResearch/PySparseLP/master/images/segmentation_lp.svg)
 
This problem can be more efficiently solved using graph-cuts but it is still interesting to compare the different generic LP solvers on this problem. 


	from pysparselp.example1 import run
	run()

segmentation with the same random data term with the optimisations limited to 15 seconds for each method
![curves](https://rawgithub.com/martinResearch/PySparseLP/master/images/potts_results.png)
convergence curves
![curves](./images/potts_curves.png)

Instead of using a simple Potts model we could try to solve the LP from [5]

## Sparse inverse convariance matrix 
 
The Sparse Inverse Covariance Estimation aims to find
a sparse matrix B that approximate the inverse of Covariance matrix A.

![latex:\large $B^*=argmin_B \|B\|_1~ s.t.~ \|A\times B-I_d\|_\infty\leq \lambda$](https://rawgithub.com/martinResearch/PySparseLP/master/images/sparse_inv_covariance.svg)

let denote f the fonction that take a matrix as an input an yield the vector of coefficient of the matrix in  row-major order.
let b=f(B) we have f(AB)=Mb with M=kron(A,I_d)
the problem rewrites

![latex: \large $ min_{b,c} \sum_i c_i ~s.t.~ -b\leq c,~b\leq c,~-\lambda\leq M b-f(I_d)\leq \lambda$](https://rawgithub.com/martinResearch/PySparseLP/master/images/lp_sparse_inv_covariance.svg)

we take inspiration from this scikit-learn example [here](http://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html) to generate 
samples of a gaussian with a sparse inverse covariance (precision) matrix. From the sample we compute the empirical covariance A and the we estimate a sparse inverse covariance (precision) matrix B from that empirical covariance using the LP formulation above.

	from pysparselp.example2 import run
	run()

![curves](./images/sparse_precision_matrix.png)

## L1 regularised multi-class SVM

Given n examples of vector-class pairs *(x_i,y_i)*, with *x_i* a vector of size m and *y_i* an integer representing the class, we aim at estimating a matrix W of size k by m that allows to discriminate the right class, with k the number of classes. We assume that the last component of *x_i* is a one in order to represent the offset constants in W. we denote *W_k* the kth line of the matrix *W*

![latex:\large $W^*=argmin_W min_{\epsilon}\|W\|_1+\sum_{i=1}^n \epsilon_i\\ s.t.~ W_{y_i}x_i-W_kx_i>1-\epsilon_i \forall\{(i,k)|k\neq y_i\}$](https://rawgithub.com/martinResearch/PySparseLP/master/images/l1svm.svg)

by adding auxiliary variables in a matrix S of the same size as the matrix W we can rewrite the absolute value as follow:
![latex:\large $\|W\|_1=min_S \sum_{ij}S_{ij} \\ s.t.~ W_{ij}<S_{ij}, -W_{ij}<S_{ij} \forall(ij)$](https://rawgithub.com/martinResearch/PySparseLP/master/images/abstolp.svg)

we obtain the LP formulation:

![latex:\large $W^*=argmin_{W} min_{\epsilon,S}  \sum_{ij}S_{ij} +\sum_{i=1}^n \epsilon_i\\s.t.~W_{y_i}x_i-W_kx_i>1-\epsilon_i \forall\{(i,k)|k\neq y_i\},W_{ij}<S_{ij}, -W_{ij}<S_{ij} \forall(ij)$](https://rawgithub.com/martinResearch/PySparseLP/master/images/l1svmLP.svg)


you can run the example using the following line in python

	from pysparselp.example3 import run
	run()


the support vectors are represented by black circles.

![classification result with support points](https://rawgithub.com/martinResearch/PySparseLP/master/images/l1svmClassification.svg)

## Bipartite matching 

Bipartite matching can be reformulated as an integer program

![latex: $$ max \sum_{ij\in \{1,\dots,n\}^2} M_{ij} C_{i,j} ~ s.t~ M_{ij}\in\{0,1\}, \sum_j M_{ij}\leq 1 \sum_i M_{ij}\leq 1 $$](./images/bipartite.svg)

we relax it into an LP.

	from pysparselp.example4 import run
	run()



## K-medians

Given n point we want to cluster them into k set by minimizing

![latex: $min_ {C \subset \{1,\dots,n\}} \sum_i min_{j\in C}d_{ij}~ s.t~ card(C)\leq k$](./images/kmedians1.svg)
with d_ij the distance between point i and point j
The can be reformulated as an integer program

![latex: $$ min \sum_{ij\in \{1,\dots,n\}^2} L_{ij} d_{ij} ~ s.t~ L_{ij}\in\{0,1\}, \sum_j L_{ij}=1 \forall i, L_{ij}<u_i \forall (i,j),\sum_i u_i\leq k $$](./images/kmedians2.svg)
 
we relax it into using 

![latex: $$ L_{ij}\in[0,1]$$](./images/kmedians2_relax.svg)
 
	from pysparselp.example5 import run
	run()

![kmedians result](./images/kmedians.svg)


## Netlib LP problems 

We have an interface to easily test the solvers on netlib problems from [netlib](http://www.netlib.org/lp/data/).  
The uncompressed files are downloaded from [here](ftp://ftp.numerical.rl.ac.uk/pub/cuter/netlib/). 
In order to monitor convergence rates, the exact solutions of these problems are found  [here](http://www.zib.de/koch/perplex/data/netlib/txt/)

	from pysparselp.test_netlib import test_netlib
	test_netlib('SC50A')

![curves](./images/libnetSC50A.png)
Note: since august 2017, numpy files containing the netlib examples are provided with scipy [here] (https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/linprog_benchmark_files)

## Random problems 

random sparse LP problem can be generate using code in *randomLP.py*. The approach used to generate random problem is very simple and could be improved
to generate harder sparse LPs. We could implement the approach used in section 6.2.1 in https://arxiv.org/pdf/1404.6770v3.pdf to 
generate random problems with the matlab code  available [here](https://github.com/YimingYAN/pipm-lp/tree/master/Tests/Ultilities)

# TODO

* document the active-set *hack* for the chambole pock method (in ChambollePockPPDAS.py).

* finish coding the method by Conda (CondatPrimalDual.py)

* convert to python the matlab implementation of the LP solver based on improved version of champolle-pock called [Adaptive Primal-Dual Hybrid Gradient Methods](https://arxiv.org/abs/1305.0546) available [here](https://www.cs.umd.edu/~tomg/projects/pdhg/)

* create a cython binding for LPsparse [1] using scipy.sparse matrices for the interface and adding the possibility to compute the convergence curve by providing the problem known solution to the solver or by adding the possibility to define a callback to a python function.

* implement method [4]

* add simplex methods written in python, could get code from here https://bitbucket.org/jbolinge/lp or speedup scipy code 
  [here](https://github.com/scipy/scipy/blob/master/scipy/optimize/_linprog.py) by getting rid of slow loops and using cython.

* try to get more meaningfull convergence curves for scipy.linprog, or maybe those are the expected curves ? 

* we provide an implementation of Mehrotra's Predictor-Corrector Pimal-Dual Interior Point method translated to python from  [Yiming yan's matlab code](https://github.com/YimingYAN/mpc). We could add other interior point methods by translating into python the code 
	* https://github.com/YimingYAN/pathfollow  (matlab)
	* https://github.com/YimingYAN/pipm-lp        (matlab)
	* http://www.cs.ubc.ca/~pcarbo/convexprog.html
	* https://github.com/YimingYAN/cppipm (c++)
	* https://github.com/pkhuong/cholesky-is-magic (lisp) described here https://www.pvk.ca/Blog/2013/12/19/so-you-want-to-write-an-lp-solver/
	  
	  
* implement some presolve methods to avoid singular matrices in the interior point methods	 (for example http://www.davi.ws/doc/gondzio94presolve.pdf). For example detect constraints on singletons, duplicated rows etc.

# Alternatives

## Linear Program solvers with a python interface
* Scipy's [linprog](http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html). Only the simplex is implemented in october 2016 (Note: an [interior point method](http://scipy.github.io/devdocs/optimize.linprog-interior-point.html) has been added in august 2017). Note that it is possible to call this solver from within our code using *method='ScipyLinProg'* when callign the *solve* method. The simplex method is implemented in python with many loops and is very slow for problems that involve more than a hundred variables. The interior point method has not been tested here.
* Python bindings for GLPK [here](https://en.wikibooks.org/wiki/GLPK/Python) . Might not be adapted to very large sparse problems as it use simplex or interior point methods. The installation is a bit tedious. The licence is GPL which makes it unsuited for use in commercial products.
* [CyLP](http://mpy.github.io/CyLPdoc/index.html) 
* [CVXOPT](http://cvxopt.org/), provides linear program cone program solvers and also provides interfaces to GLPK,Mosek,DSPD. 
* [CVXPY](http://www.cvxpy.org/en/latest/) Python-embedded modeling language for convex optimization problems. It provide interface to cvxopt solvers and to SCS
* [SCS](https://pypi.python.org/pypi/scs), [github](https://github.com/cvxgrp/scs) Solves convex cone programs via operator splitting. Can solve in particular linear programs. 
## No python interface

* [LIPSOL](http://www.caam.rice.edu/~zhang/lipsol/). Seems to be adequate for sparse problems. Part of the code in fortran. licence GPL
* [LPsolve](https://sourceforge.net/projects/lpsolve/) licence LGPL
* [Joptimize](http://www.joptimizer.com/linearProgramming.html) appache licence
* [PCx](http://pages.cs.wisc.edu/~swright/PCx/) PCx is an interior-point predictor-corrector linear programming package. Code available here https://github.com/lpoo/PCx. Free but to public domain
* [DSDP](http://www.mcs.anl.gov/hs/software/DSDP/) solve semidefinite programs, which are more general than linear programs. It uses the sparsity of the problem and might still be competitive to solve sparse linear programs. Can be called from python through [cvxopt](http://cvxopt.org/)

# References

[1] Ian En-Hsu Yen,  Kai Zhong,  Cho-Jui Hsieh, Pradeep K Ravikumar, Inderjit S Dhillon *Sparse Linear Programming via Primal and Dual Augmented Coordinate Descent*, NIPS 2015

[2] T. Pock and A.Chambolle *Diagonal preconditioning for first order primal-dual algorithms in convex optimization*  ICCV 2011

[3]  Stephen Boyd *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*  Foundations and Trends in Machine Learning 2010

[4] Yu G Evtushenko, A I Golikov, and N Mollaverdy. *Augmented
Lagrangian method for large-scale linear programming problems*  Optimization Method and Software 2005.