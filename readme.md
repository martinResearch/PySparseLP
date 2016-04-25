# Goal

This projet provides several python codes to solve sparse linear programs of the form

![latex:\Large $ \mathbf{x}^*=argmin_x \mathbf{c}^t\mathbf{x} ~  s.t.~  A_e\mathbf{x}=\mathbf{b_e},A_i\mathbf{x}\leq\mathbf{ b_i}, \mathbf{l}\leq \mathbf{x}\leq \mathbf{u}$ ](./images/LPproblem.svg)


The differents provided algorithms are 

* a dual coordinate ascent method with exact line search 
* a dual gradient ascent with exact line search
* a first order primal-dual algorithm adapted from chambolle pock [2]
* two methods based on the Alternating Direction Method of Multipliers [3]

# installation


using pip

	sudo pip install -e hg+https://bitbucket.org/martin_delagorce/pysparselp#egg=pysparselp

# Examples

## image sementation
we can use it to solve a binary image segmentation problem with Potts regularization.
This problem can be more efficiently solved using graph cut but it is of interest to compare different generic LP solver on this problem. 

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
	* examples from the paper using LP for imge processing ? 

* document th eactive set hack for the chambole pock method ? 


* ceate a cython binding for LPsparse [] and add the possibility to comput the convergence curve 



[1]*Sparse Linear Programming via Primal and Dual Augmented Coordinate Descent*
Ian En-Hsu Yen,  Kai Zhong,  Cho-Jui Hsieh, Pradeep K Ravikumar, Inderjit S Dhillon, NIPS 2015

[2] *Diagonal preconditioning for first order primal-dual algorithms in convex optimization* T. Pock and A.Chambolle ICCV 2011

[3] *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers* Stephen Boyd Foundations and Trends in Machine Learning 2010

		

