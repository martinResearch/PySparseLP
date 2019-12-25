import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from pysparselp.SparseLP import SparseLP,solving_methods
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse
import scipy.ndimage
import scipy.signal
import sys
#from imageio import imwrite
import  maxflow #pip install PyMaxflow
import os


class ImageLP(SparseLP):



	def addPenalizedDifferences(self,I,J,coefpenalization):
		assert(I.size==J.size)
		maxDiff=np.maximum(self.upperbounds[I]-self.lowerbounds[J],self.upperbounds[J]-self.lowerbounds[I])
		aux=self.addVariablesArray(I.shape,upperbounds=maxDiff,lowerbounds=0,costs=coefpenalization)
		if np.isscalar(coefpenalization):
			assert(coefpenalization>0)
		else:#allows a penalization that is different for each edge (could be dependent on an edge detector)
			assert(coefpenalization.shape==aux.shape)
			assert(np.min(coefpenalization)>=0)
		aux_ravel=aux.ravel()
		I_ravel=I.ravel()
		J_ravel=J.ravel()
		cols=np.column_stack((I_ravel,J_ravel,aux_ravel))
		vals=np.tile(np.array([1,-1,-1]),[I.size,1])
		self.addLinearConstraintRows(cols,vals,lowerbounds=None,upperbounds=0)
		vals=np.tile(np.array([-1,1,-1]),[I.size,1])
		self.addLinearConstraintRows(cols,vals,lowerbounds=None,upperbounds=0)

	def addPottHorizontal(self,indices,coefpenalization):
		self.addPenalizedDifferences(indices[:,1:],indices[:,:-1],coefpenalization)
	def addPottVertical(self,indices,coefpenalization):
		self.addPenalizedDifferences(indices[1:,:],indices[:-1,:],coefpenalization)
	def addPottModel(self,indices,coefpenalization):
		self.addPottHorizontal(indices,coefpenalization)
		self.addPottVertical(indices,coefpenalization)



def run():
	thisfilepath=os.path.dirname(os.path.abspath(__file__))
	plt.ion()


	nLabels = 1
	np.random.seed(1)
	imageSize=50
	coefPotts=0.5
	size_image=(imageSize,imageSize,nLabels)
	nb_pixels=size_image[0]*size_image[1]
	coefMul=500# we multiply all term by theis constant because the graph cut algorithm take integer weights.
	unary_terms=np.round(coefMul*((np.random.rand(size_image[0],size_image[1],size_image[2]))*2-1))
	coefPotts=round(coefPotts*coefMul)



	H=unary_terms.shape[0]
	W=unary_terms.shape[1]
	g = maxflow.Graph[int](0, 0)
	nodeids = g.add_grid_nodes(unary_terms.shape)

	alpha=coefPotts
	g.add_grid_edges(nodeids, alpha)
	# Add the terminal edges.
	g.add_grid_tedges(nodeids, unary_terms*0, unary_terms)

	print ("calling maxflow")
	g.maxflow()
	sgm = g.get_grid_segments(nodeids)
	img2 = np.int_(np.logical_not(sgm))
	plt.imshow(img2[:,:,0], cmap=plt.cm.gray, interpolation='nearest')
	plt.show()

	LP=ImageLP()

	indices=LP.addVariablesArray(shape=size_image,lowerbounds=0,upperbounds=1,costs=unary_terms/coefMul)

	groundTruth=img2
	groundTruthIndices=indices

	LP.addPottModel(indices, coefpenalization=coefPotts/coefMul)

	print ("solving")

	fig_solutions=plt.figure()

	im=plt.imshow(unary_terms[:,:,0]/coefMul,cmap=plt.cm.Greys_r,interpolation="nearest",vmin=0,vmax=1)
	fig_curves=plt.figure()
	ax_curves1=plt.gca()
	ax_curves1.set_xlabel('nb of iteration')
	ax_curves1.set_ylabel('distanceToGroundTruth')
	fig_curves=plt.figure()
	ax_curves2=plt.gca()
	ax_curves2.set_xlabel('duration')
	ax_curves2.set_ylabel('distanceToGroundTruth')	
	def plotSolution(niter,solution,is_active_variable=None):
		image=solution[indices]
		#imwrite('ter%05d.png'%niter,solution[indices][:,:,0])
		#imwrite('diff_iter%05d.png'%niter,np.diff(solution[indices][:,:,0]))
		im.set_array(image[:,:,0])
		#im.set_array(np.diff(image[:,:,0]))
		plt.draw()
		plt.show()



	fig=plt.figure()
	ax=fig.add_subplot(2,5,1,title='graph cut')
	ax.imshow(groundTruth[:,:,0],cmap=plt.cm.Greys_r,interpolation='none')
	ax.axis('off')


	# simplex much too slow for images larger than 20 by 20
	#LP2=copy.deepcopy(LP)
	#LP2.convertToOnesideInequalitySystem()
	#sol1,elapsed=LP2.solve(method='ScipyLinProg',force_integer=False,getTiming=True,nb_iter=100,max_time=10,groundTruth=groundTruth,groundTruthIndices=indices,plotSolution=None)
	
	solving_methods2=[m for m in solving_methods if (not m in ['ScipyLinProg'])] # remove ScipyLinProg because it is too slow
	for i,method in enumerate(solving_methods2):
		print('\n\n----------------------------------------------------------\nSolving LP using %s'%method)
		sol1,elapsed=LP.solve(method=method,getTiming=True,nb_iter=1000000,max_time=15,groundTruth=groundTruth,groundTruthIndices=indices,plotSolution=None,nb_iter_plot=500)
		ax_curves1.semilogy(LP.itrn_curve,LP.distanceToGroundTruth,label=method)
		ax_curves2.semilogy(LP.opttime_curve,LP.distanceToGroundTruth,label=method)
		ax_curves1.legend()
		ax_curves2.legend()
		ax=fig.add_subplot(2,5,i+2,title=method)
		ax.imshow(sol1[indices][:,:,0],cmap=plt.cm.Greys_r,interpolation='none',vmin=0,vmax=1)
		ax.axis('off')
		plt.draw()
		plt.show()
	plt.tight_layout()
	#plt.figure()
	#plt.plot(LP.itrn_curve,LP.dopttime_curve,'g',label='ADMM')
	#plt.draw()
	#plt.show()
	print ('done')

if __name__ == "__main__":
	run()



