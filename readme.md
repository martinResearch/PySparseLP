# Goal

This projet provides several python codes to solve sparse linear programs of the form

![latex:\large $\mathbf{x}^*=argmin_\mathbf{x} \mathbf{c}^t\mathbf{x} ~  s.t.~  A_e\mathbf{x}=\mathbf{b_e},A_i\mathbf{x}\leq\mathbf{ b_i}, \mathbf{l}\leq \mathbf{x}\leq \mathbf{u}$ ](./images/LPproblem.svg)


The differents algorithms that are implemented are documented in the [pdf](./latex/SparseLinearProgramming.pdf): 

* a dual coordinate ascent method with exact line search 
* a dual gradient ascent with exact line search
* a first order primal-dual algorithm adapted from chambolle pock [2]
* three methods based on the Alternating Direction Method of Multipliers [3]

**Note** These methods are not meant to be efficient methods to solve generic linear programs. They are simple and quite naive methods i coded while exploring different possibilites to solve sparse linear programs.


This project also provides: 

* a python class that  make is easier to building LP programs from python, 
* methods to convert between the different common forms of LP (slack form , standard form etc), 
* methods to export to standard file formats (MPS)

# Installation


using pip

	sudo pip install -e hg+https://bitbucket.org/martin_delagorce/pysparselp#egg=pysparselp

# Examples

## Image segmentation
we can use it to solve a binary image segmentation problem with Potts regularization.

![latex: \large $min_s c^ts + \sum_{(i,j)\in E}  |s_i-s_j| ~s.t. ~0 \leq s\leq 1$](./images/segmentation.svg)

with E the list of indices of pairs of neighboring pixels and c a cost vector that is obtain from color distribution models of the two regions.
This poblem can be rewritten as a linear progamme by adding an auxilay variable d_ij for each edge with the constraints

![latex: \large $min_s c^ts + \sum_{(i,j)\in E}  d_{ij} ~s.t. ~0 \leq s\leq 1, ~d_{ij}\geq s_j-s_j, ~d_{ij}\geq s_i-s_i $](./images/segmentation_lp.svg)
 
This problem can be more efficiently solved using graph-cuts but it is still interesting to compare the different generic LP solvers on this problem. 


	import pysparselp.example1

segmentation with the same random data term with the optimizations limited to 15 seconds for each method
![curves](./images/potts_results.png)
convergence curves
![curves](./images/potts_curves.png)

Instead of using a simple Potts model we could try to solve the LP from [5]

## Sparse inverse convariance matrix 
 
The Sparse Inverse Covariance Estimation aims to find
a sparse matrix B that approximate the inverse of Covariance matrix A.

![latex:\large $B^*=argmin_B \|B\|_1~ s.t.~ \|A\times B-I_d\|_\infty\leq \lambda$](./images/sparse_inv_covariance.svg)

let denote f the fonction that take a matrix as an input an yield the vector of coefficient of the matrix in  row-major order.
let b=f(B) we have f(AB)=Mb with M=kron(A,I_d)
the problem rewrites

![latex: \large $ min_{b,c} \sum_i c_i ~s.t.~ -b\leq c,~b\leq c,~-\lambda\leq M b-f(I_d)\leq \lambda$](./images/lp_sparse_inv_covariance.svg)

we take inspiration from this scikit-learn example [here](http://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html) to generate 
samples   of a gaussian with a sparse inverse covariance (precision) matrix. From the sample we comute the empirical covariance and the we estimate a sparse inverse covariance (precision) matrix from sample using the LP formulation above.

	import pysparselp.example2


![curves](./images/sparse_precision_matrix.png)

## Other problems
test data can be obtained from  [netlib.org](http://www.netlib.org/lp/data/)

but we need to write

* a cython wrapper for the code that decompress emps files to mps (http://www.netlib.org/lp/data/emps.c)
* a code to read MPS file in python



#TODO

* add more examples 
	* sparse inverse covariance matrix (see [1])
	* L1 SVM (see [1])
	* examples from the paper using LP for image processing ? 

* document the active-set *hack* for the chambole pock method.

* create a cython binding for LPsparse [1] using scipy.sparse matrices for the interface and adding the possibility to compute the convergence curve by providing the problem known solution to the solver.

* implement method [4]

# References

[1] Ian En-Hsu Yen,  Kai Zhong,  Cho-Jui Hsieh, Pradeep K Ravikumar, Inderjit S Dhillon *Sparse Linear Programming via Primal and Dual Augmented Coordinate Descent*, NIPS 2015

[2] T. Pock and A.Chambolle *Diagonal preconditioning for first order primal-dual algorithms in convex optimization*  ICCV 2011

[3]  Stephen Boyd *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*  Foundations and Trends in Machine Learning 2010

[4] Yu G Evtushenko, A I Golikov, and N Mollaverdy. *Augmented
Lagrangian method for large-scale linear programming problems*  2005.

[5] *A linear framework for region-based image segmentation
and inpainting involving curvature penalization*
