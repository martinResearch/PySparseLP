# kmedians LP relaxation example
# LP formulation inspired from https://cseweb.ucsd.edu/~dasgupta/291-geom/kmedian.pdf,
#                              http://papers.nips.cc/paper/3478-clustering-via-lp-based-stabilities.pdf
#                              https://www.cs.princeton.edu/courses/archive/fall14/cos521/projects/kmedian.pdf

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


def kmediansContraint(LP,indices):
	cols=indices		
	vals=np.ones(cols.shape)
	LP.addLinearConstraintRows(cols,vals,lowerbounds=1,upperbounds=1)	
	cols=indices.T
	vals=np.ones(cols.shape)
	LP.addLinearConstraintRows(cols,vals,lowerbounds=-np.inf,upperbounds=1)
	
def clusterize(points,k,nCenterCandidates):
	
	n=points.shape[0]
	centerCandidates=points[np.random.choice(n,nCenterCandidates),:]
	
	
	pairdistances=np.sqrt(np.sum((points[:,None,:]-centerCandidates[None,:,:])**2,axis=2))
	
	LP=SparseLP()
	labeling=LP.addVariablesArray(pairdistances.shape,0,1, pairdistances)
	
	usedAsCenter=LP.addVariablesArray(nCenterCandidates,0,1, 0)
	LP.addLinearConstraintRow(usedAsCenter,np.ones((nCenterCandidates)),lowerbound=0,upperbound=k)
	LP.addLinearConstraintRows(labeling,np.ones((n,nCenterCandidates)),lowerbounds=1,upperbounds=1)
	# max(labeling,axis=0)<usedAsCenter
	# the binary variable associated to each  column should be greater than all binary variables on that row
	id_cols=np.ones((n,1)).dot(usedAsCenter[None,:])
	cols=np.column_stack((labeling.reshape(-1,1),id_cols.reshape(-1,1)))
	vals=np.column_stack((np.ones((n*nCenterCandidates)),-np.ones((n*nCenterCandidates))))
	LP.addLinearConstraintRows(cols,vals,lowerbounds=None,upperbounds=0)		
	
	
	
	s=LP.solve(method='ADMM',nb_iter=60000,max_time=20,nb_iter_plot=500)[0]
	
	print (LP.costsvector.dot(s))
	x=s[labeling]
	print(np.round(x*1000)/1000)
	
	LP.maxConstraintViolation(s)
	
	label=np.argmax(x,axis=1)
	if not(len(np.unique(label))==k):
		print('failed')
	return label
	
def run():
	
	k=5
	n=500
	prng = np.random.RandomState(0)
	centers=prng.randn(k,2)
	gtlabel=np.floor(prng.rand(n)*5).astype(np.int)
	points=0.4*prng.randn(n,2)+centers[gtlabel,:]
	plt.ion()
	plt.plot(points[:,0],points[:,1],'.')	
	plt.draw()	
	plt.show()	
	nCenterCandidates=50
	
	label=clusterize(points,k,nCenterCandidates)
	
	for i in np.arange(n):
		if any(label==i):
			plt.plot(points[label==i,0],points[label==i,1],'o')
	plt.draw()	
	plt.show()
	plt.axis('equal')
	plt.tight_layout()
	print ('done')
	
if __name__ == "__main__":
	run()

	
	
