# Goal

This projet provides several python codes to solve sparse linear programs of the form

![latex:\large $ \mathbf{x}^*=argmin_\mathbf{x} \mathbf{c}^t\mathbf{x} ~  s.t.~  A_e\mathbf{x}=\mathbf{b_e},A_i\mathbf{x}\leq\mathbf{ b_i}, \mathbf{l}\leq \mathbf{x}\leq \mathbf{u}$ ](./images/LPproblem.svg)


The differents algorithms that are implemented are 

* a dual coordinate ascent method with exact line search 
* a dual gradient ascent with exact line search
* a first order primal-dual algorithm adapted from chambolle pock [2]
* three methods based on the Alternating Direction Method of Multipliers [3]

**Note** This method are not meant to be efficient method to solve generic linear programs. They are simple and quite naive methods i coded while exploring different possibilites to solve sparse linear programs.


This project also provides: 
* a python class that  make is easier to building LP programs from python, 
* methods to convert between the different common forms of LP (slack form , standard form etc), 
* methods to export to standard file formats (MPS)

# Installation


using pip

	sudo pip install -e hg+https://bitbucket.org/martin_delagorce/pysparselp#egg=pysparselp

# Examples

## image sementation
we can use it to solve a binary image segmentation problem with Potts regularization.
This problem can be more efficiently solved using graph-cuts but it is still interesting to compare the different generic LP solvers on this problem. 

## Sparse inverse convariance matrix 
 


##
test data can be obtained from  
http://www.netlib.org/lp/data/
but need 
- to write a cython wrapper for the code that decompress emps files to mps (http://www.netlib.org/lp/data/emps.c)
- to write code to read MPS file in python



#TODO

* add more examples 
	* sparse inverse covariance matrix (see [])
	* L1 SVM 
	* examples from the paper using LP for image processing ? 

* document the active set *hack* for the chambole pock method.


* ceate a cython binding for LPsparse [1] using scipy.sparse matrices for the interface and adding the possibility to compute the convergence curve yt providing the problem solution to the solvr to compute error curves.

## References

[1] *Sparse Linear Programming via Primal and Dual Augmented Coordinate Descent*
Ian En-Hsu Yen,  Kai Zhong,  Cho-Jui Hsieh, Pradeep K Ravikumar, Inderjit S Dhillon, NIPS 2015

[2] *Diagonal preconditioning for first order primal-dual algorithms in convex optimization* T. Pock and A.Chambolle ICCV 2011

[3] *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers* Stephen Boyd Foundations and Trends in Machine Learning 2010

		

