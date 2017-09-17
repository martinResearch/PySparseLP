# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------
#Copyright © 2016 Martin de la Gorce <martin[dot]delagorce[hat]gmail[dot]com>

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------
import numpy as np

def conjgrad(A,b,x0,maxiter=100,tol=1e-10):
	""" This function solves Ax=b using the conjugate gradient method
	    converted from the matlab code from wikipedia
	    wikipedia http://en.wikipedia.org/wiki/Conjugate_gradient_method
	"""
	x=x0.copy()
	r=b-A*x
	p=r
	rsold=r.dot(r)
	
	for i in range(maxiter):
		Ap=A*p
		alpha=rsold/(p.dot(Ap))
		x=x+alpha*p
		r=r-alpha*Ap
		rsnew=r.dot(r)
		if np.sqrt(rsnew)<tol:
			break
			
		p=r+(rsnew/rsold)*p;
		rsold=rsnew
	return x 
