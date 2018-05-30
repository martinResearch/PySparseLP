#cython: profile=True
#cython: boundscheck=False    
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True

import numpy

#cython#cimport libc.math  as math
#cython#cimport numpy as np
#cython#cimport cython

#cython#cdef\
class xorshift:
        # simple pseudo-random generator base on the xorshift algorithm
        # the goal is to get a random generator that can easily be reimpleted in various languages(matlab for example) in order to be able to generate
        # the same random number sequences in various languages
        # http://stackoverflow.com/questions/5829499/how-to-let-boostrandom-and-matlab-produce-the-same-random-numbers
        # http://stackoverflow.com/questions/3722138/is-it-possible-to-reproduce-randn-of-matlab-with-numpy        
        
        #cython#cdef unsigned long x,y,z,w
        #cython#cdef float max
        
        def __init__(self):
                
                self.x = 123456789
                self.y = 362436069
                self.z = 521288629
                self.w = 88675123
                self.max=2**32
        def next(self):
                t = self.x ^ (self.x<<11)& 0xffffffff                   # <-- keep 32 bits
                self.x = self.y
                self.y = self.z
                self.z = self.w
                w = self.w
                self.w = w ^ (w >> 19) ^(t ^ (t >> 8))& 0xffffffff                   # <-- keep 32 bits
                return self.w
        def rand(self,m=1,n=1):
                #cython#cdef np.ndarray[np.float64_t, ndim=2]\
                r=numpy.empty((m,n))               
                for i in range(m):
                        for j in range(n):
                                r[i,j]=float(self.next())/self.max
                return r
            
        def randint(self,a,b):            
                r=int(a+(b-a+1)*self.rand())
                return r
        
        
        def choice( self,set):
                i=self.randint(0,len(set)-1)
                return set[i]
            
        def randn(self,m=1,n=1):
                return self.normal(0,1,m=m,n=n)
        
        def normal(self,mean,std,m=1,n=1):
                # generate nromal distributed pseudo random numbers using the box-muller transform
                # http://en.wikipedia.org/wiki/Box-Muller_transform
                u1=self.rand(m,n)
                u2=self.rand(m,n)
                return mean+std*numpy.sqrt(-2*numpy.log(u1))*numpy.cos(2*numpy.pi*u2)