# maximum bipartite matching example

import matplotlib.pyplot as plt
import time
import copy
from SparseLP import SparseLP,solving_methods
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse
import scipy.ndimage
import scipy.signal
import sys
import os


def addBipartiteContraint(LP,indices):
	cols=indices		
	vals=np.ones(cols.shape)
	LP.addLinearConstraintRows(cols,vals,lowerbounds=-np.inf,upperbounds=1)	
	cols=indices.T
	vals=np.ones(cols.shape)
	LP.addLinearConstraintRows(cols,vals,lowerbounds=-np.inf,upperbounds=1)

def run():
	
	n=50
	np.random.seed(2)
	Cost=-np.random.rand(n,n)
	LP=SparseLP()
	indices=LP.addVariablesArray(Cost.shape,0,1, Cost)
	addBipartiteContraint(LP,indices)
	s=LP.solve(method='Mehrotra',nb_iter=60000,max_time=20)[0]
	print (LP.costsvector.dot(s))
	s=LP.solve(method='DualCoordinateAscent',nb_iter=60000,max_time=40,nb_iter_plot=500)[0]
	s=LP.solve(method='ChambollePockPPD',nb_iter=60000,max_time=10,nb_iter_plot=500)[0]
	x=s[indices]
	print(np.round(x*1000)/1000)
	print ('done')
	
if __name__ == "__main__":
	run()

	
	
