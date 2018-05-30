import numpy as np
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
from scipy.misc import imsave
import  maxflow #pip install PyMaxflow 
import os
from netlib import getProblem
import urllib
import os.path
import gzip
import zlib

def test_netlib(pbname):
	
	plt.ion()
	LPDict=getProblem(pbname)
	groundTruth=LPDict['solution']

		
	LP=SparseLP()
	nbvar=len(LPDict['costVector'])
	LP.addVariablesArray(nbvar,lowerbounds=LPDict['lowerbounds'],
                                   upperbounds=LPDict['upperbounds'],
                                   costs=LPDict['costVector'])
	LP.addEqualityConstraintsSparse(LPDict['Aeq'],LPDict['Beq'])
	LP.addConstraintsSparse(LPDict['Aineq'],LPDict['B_lower'],LPDict['B_upper'])
	
	print "solving"

	f, axarr = plt.subplots(3, sharex=True)
	axarr[0].set_title('mean absolute distance to solution')
	axarr[1].set_title('maximum constraint violation')
	axarr[2].set_title('difference with optimum value')
	
	LP2=copy.deepcopy(LP)
	LP2.convertToOnesideInequalitySystem()
	
	#LP2.saveMPS(os.path.join(thisfilepath,'data','example_reexported.mps'))
	
	LP=LP2
	assert(LP.checkSolution(groundTruth))
	costGT=LP.costsvector.dot(groundTruth.T)

	
	scipySol,elapsed=LP2.solve(method='ScipyLinProg',force_integer=False,getTiming=True,nb_iter=100000)
	max_time=3
	method='ScipyLinProg'
	if not scipySol is np.nan:
		sol1=scipySol
		maxv=LP.maxConstraintViolation(sol1)
		# compute the primal and dual infeasibility 
		print '%s found  solution with maxviolation=%2.2e and  cost %f (vs %f for ground truth) in %f seconds'%(method,maxv,LP.costsvector.dot(sol1),costGT,elapsed)
		print 'mean of absolute distance to gt solution =%f'%np.mean(np.abs(groundTruth-sol1))	
	else:
		print 'scipy simplex did not find a solution'
		

	# testing our methods
	
	solving_methods2=[m for m in solving_methods if (not m in ['ScipyLinProg','DualCoordinateAscent'])]
	#solving_methods2=['Mehrotra']
	for i,method in enumerate(solving_methods2):
		sol1,elapsed=LP.solve(method=method,force_integer=False,getTiming=True,nb_iter=1000000,max_time=max_time,groundTruth=groundTruth,plotSolution=None)
		axarr[0].semilogy(LP.opttime_curve,LP.distanceToGroundTruth,label=method)
		axarr[1].semilogy(LP.opttime_curve,LP.max_violated_constraint)
		axarr[2].semilogy(LP.opttime_curve,LP.pobj_curve-costGT)
		axarr[0].legend()
		plt.show()
	print 'done'
	
if __name__ == "__main__":
	
	#test_netlib('afiro')# seems like the solution is not unique
	test_netlib('SC50B')
	#test_netlib('SC50A')
	#test_netlib('KB2')
	#test_netlib('SC105')
	#test_netlib('ADLITTLE')# seems like the solution is not unique
	#test_netlib('SCAGR7')
	#test_netlib('PEROLD')# seems like there is a loading this problem 
	#test_netlib('AGG2')
	
	
	
	
	

	
	
