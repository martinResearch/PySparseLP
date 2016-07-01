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

class L1SVM(SparseLP):
	""" L1-regularized multi-class Support	Vector Machine  J. Zhu, S. Rosset, T. Hastie, and R. Tibshirani. 1-norm support vector machines. NIPS, 2004."""
	
	def addAbsPenalization(self,indices,coefpenalization):
		
		aux=self.addVariablesArray(indices.size,upperbounds=None,lowerbounds=0)
		
		if np.isscalar(coefpenalization):
			assert(coefpenalization>0)
			self.setCostsVariables(aux, np.ones(aux.shape)*coefpenalization)
		else:#allows a penalization that is different for each edge (could be dependent on an edge detector)
			assert(coefpenalization.shape==aux.shape)
			assert(np.min(coefpenalization)>=0)
			self.setCostsVariables(aux, np.ones(aux.shape)*coefpenalization)
	
		#start by adding auxilary variables 
	
		aux_ravel=aux.ravel()
		indices_ravel=indices.ravel()
		cols=np.column_stack((indices_ravel,aux_ravel))
		vals=np.tile(np.array([1,-1]),[indices.size,1])
		self.addLinearConstraintRows(cols,vals,lowerbounds=None,upperbounds=0)
		vals=np.tile(np.array([-1,-1]),[indices.size,1])
		self.addLinearConstraintRows(cols,vals,lowerbounds=None,upperbounds=0)		
			
	
	def setData(self,x,classes,nbClasses=None):
		nbExamples=x.shape[0]
		xh=np.hstack((x,np.ones((nbExamples,1))))
		assert(x.shape[0]==len(classes))
		if nbClasses is None:
			nbClasses=np.max(classes)+1
		nbFeatures=x.shape[1]
		
		self.weightsIndices=self.addVariablesArray((nbClasses,nbFeatures+1),None,None)
		self.addAbsPenalization(self.weightsIndices,1)
		self.epsilonsIndices=self.addVariablesArray((nbExamples,1),upperbounds=None,lowerbounds=0,costs=1)
		e=np.ones((nbExamples,nbClasses))
		e[np.arange(nbExamples),classes]=0
		
		#sum(x*weights[classes,:]),axis=1)[:,None]- x.dot(weights)+epsilon>e
		
		cols1 = self.weightsIndices[classes,:]
		vals1 = xh
		for k in range(nbClasses):
			keep=classes!=k
			cols2 = np.tile(self.weightsIndices[[k],:],[nbExamples,1])
			vals2 = -xh		
			vals3 = np.ones(self.epsilonsIndices.shape)
			cols3 = self.epsilonsIndices
			vals  = np.column_stack((vals1,vals2,vals3))
			cols  = np.column_stack((cols1,cols2,cols3))
			self.addLinearConstraintRows(cols[keep,:],vals[keep,:],lowerbounds=e[keep,k],upperbounds=None)	
		
	def train(self,method='ADMM2'):
		sol1,elapsed=self.solve(method=method,force_integer=False,getTiming=True,nb_iter=1000000,max_time=5,plotSolution=None)
		self.weights=sol1[self.weightsIndices]
		marges=sol1[self.epsilonsIndices]
		self.activeSet=np.nonzero(marges>1e-3)[0]
	
	def classify(self,x):
		nbExamples=x.shape[0]
		xh=np.hstack((x,np.ones((nbExamples,1))))
		scores=xh.dot(self.weights.T)
		classes=np.argmax(scores,axis=1)
		return classes
		

def run():
	plt.ion()
	

	
	nLabels = 1  
	np.random.seed(1)
	nbClasses=3
	nbExamples=1000
	x=np.random.rand(nbExamples,2)
	xh=np.hstack((x,np.ones((nbExamples,1))))
	#plt.plot(x[:,0],x[:,1],'.')
	
	weights=np.random.randn(nbClasses,2)
	weights=weights/np.sum(weights**2,axis=1)[:,None]
	weights=np.hstack((weights,-0.5*np.sum(weights,axis=1)[:,None]))
	scores=(weights.dot(xh.T)).T
	classes=np.argmax(scores,axis=1)
	
	colors=['r','g','b']
	plt.ion()
	for k in range(3):
		plt.plot(x[classes==k,0],x[classes==k,1],'.',color=colors[k])
		
		
	l1svm=L1SVM()
	l1svm.setData(x,classes)
	l1svm.train()
	classes2=l1svm.classify(x)
	
	colors=['r','g','b']
	plt.figure()
	plt.ion()
	for k in range(3):
		plt.plot(x[classes2==k,0],x[classes2==k,1],'.',color=colors[k])
	plt.plot(x[l1svm.activeSet,0],x[l1svm.activeSet,1],'ko',markersize=10,fillstyle='none' )
	plt.axis('equal')
	plt.axis('off')
	plt.ioff()
	plt.show()
	print 'done'	



if __name__ == "__main__":
	run()


		
	
	
	
