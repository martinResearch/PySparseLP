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
from scipy import sparse
import gzip
import os
import urllib

def MPSParser(f,fsol=None):
    """
    Parse Linear programs in the MPS format. 
    This file format is described here
    https://en.wikipedia.org/wiki/MPS_(format)
    This has been coded in a rush and might not handle all cases
    could have a look at mps2mat.f that is a part of the package LIPSOL by Yin Zhang
    or https://github.com/YimingYAN/cppipm/blob/master/src/mpsReader.cpp
    """
    
    
    nb_ineq=0
    nb_eq=0
    nb_var=0
    B_lower=dict()
    B_upper=dict()
    Beq=dict()
    rows=dict()
    variables=dict()
    AineqList=[]
    AeqList=[]
    vidtovar=dict()
    partparsing=None
    while True:
       
        line=f.readline()
        line = line.rstrip('\n')
        line+=' '*(len(line)-61)
        t=[]
        t.append(line[1:3].strip())
        t.append(line[4:12].ljust(8))
        t.append(line[14:22])
        t.append(line[25:36].strip())
        t.append(line[39:47])
        t.append(line[49:61].strip())
        #t=line.split()  
        #t[0]=
        #2-3        5-12      15-22     25-36     40-47     50-61
        if len(t)==0:
            continue
        if line.startswith('ENDATA'):
            break
        if line.startswith('*'):# this is a comment
                continue 
        if len(t)==0:
            continue
        if line.startswith('NAME'):
            problem_name=t[1]
            continue
        if line.startswith('ROWS'):
            partparsing='ROWS'
            continue
        if line.startswith('COLUMNS'):
            partparsing='COLUMNS'
            continue   
        if line.startswith('RHS'):
            partparsing='RHS'
            continue  
        if line.startswith('BOUNDS'):
            partparsing='BOUNDS'
            continue  
        
        if line.startswith('RANGES'):
            print ('not coded yet')
            raise
        

        
        if partparsing=='ROWS':
            if t[0]=='N':
                costname=t[1]
               
            if t[1] in rows:
                raise
            r=dict()
            rows[t[1]]=r
            r['type']=t[0]
            if  t[0]=='G':                    
                r['id']=nb_ineq
                B_lower[nb_ineq]=0
                B_upper[nb_ineq]=np.inf
                nb_ineq+=1
                
            if t[0]=='L' :    
                r['id']=nb_ineq
                B_lower[nb_ineq]=-np.inf
                B_upper[nb_ineq]=0 
                nb_ineq+=1
            elif t[0]=='E':               
                r['id']=nb_eq
                Beq[nb_eq]=0 # set default value
                nb_eq+=1  
                
            continue
                
                
        if partparsing=='COLUMNS':
            
            if t[1] in variables:
               
                var=variables[t[1]]
            else:
                var=dict()
                variables[t[1]]=var
                var['id']=nb_var
                var['UP']=np.inf
                var['LO']=0 #Variables not mentioned in a given BOUNDS set are taken to be non-negative (lower bound zero, no upper bound)
                var['cost']=0
                vidtovar[nb_var]=var
                nb_var+=1
                
            j=var['id']
            for k in range(int((len(t)-2)/2)):
                if t[2*k+2]=='':
                    break
                r=rows[t[2*k+2]]
                v=float(t[2*k+3])
                if r['type']=='N':
                    var['cost']=v  
                    continue

                i=r['id']
               
               
                if r['type']=='L':
                    AineqList.append((i,j,v))
                elif r['type']=='G':
                    AineqList.append((i,j,v)) 
                elif r['type']=='E':
                    AeqList.append((i,j,v)) 
            continue
                    
                
        if partparsing=='RHS':  
            
            for k in range(int((len(t)-2)/2)):
                if t[2*k+2]=='':
                    break
                r=rows[t[2*k+2]]
                i=r['id']
                v=float(t[2*k+3])
                if r['type']=='N':
                    raise
                elif r['type']=='L':
                  
                    B_upper[i]=v
                elif r['type']=='G':
                    B_lower[i]=v
                    
                elif r['type']=='E':
                    Beq[i]=v  
            continue
            
            
            
        if partparsing=='BOUNDS':
            var=variables[t[2]]
            var['name']=t[2]
            if t[0]=='UP' or t[0]=='LO':
                var[t[0]]=float(t[3])
            elif t[0]=='FR':
                var['UP']=np.inf
                var['LO']=-np.inf
            elif t[0]=='FX':
                var['UP']=float(t[3])
                var['LO']=float(t[3])   
            elif t[0]=='MI':  
                var['LO']=-np.inf
            elif t[0]=='PL':  
                var['UP']=np.inf  
            elif t[0]=='BV' or t[0]=='LI' or t[0]=='UI':  
                print ('integer constraints ignored')
                raise
        
    costVector =np.array([vidtovar[i]['cost'] for i in range(nb_var)])
    upperbounds=np.array([vidtovar[i]['UP'] for i in range(nb_var)])
    lowerbounds=np.array([vidtovar[i]['LO'] for i in range(nb_var)])
    
    
    
    Aineq=sparse.dok_matrix((nb_ineq,nb_var))
    for i,j,v in AineqList:
        Aineq[i,j]=v
      
    Aeq=sparse.dok_matrix((nb_eq,nb_var))
    for i,j,v in AeqList:
        Aeq[i,j]=v        
    
    Beq=np.array([Beq[i] for i in range(nb_eq)])
    B_lower=np.array([B_lower[i] for i in range(nb_ineq)])
    B_upper=np.array([B_upper[i] for i in range(nb_ineq)])

    #print Aeq
    r={'costVector':costVector,
            'upperbounds':upperbounds,
            'lowerbounds':lowerbounds,
            'Aeq':Aeq,
            'Beq':Beq,
            'Aineq':Aineq,
            'B_lower':B_lower,
            'B_upper':B_upper}
               
   
    #parses Linear Program solution file generated by perPlex Version 1.00 
    #examples of such file in http://www.zib.de/koch/perplex/data/netlib/txt/ 
    #paper here https://opus4.kobv.de/opus4-zib/files/727/ZR-03-05.pdf
    r['solution']=None
    if not fsol is None:
        
        while True:
           
            line=fsol.readline()
            line = line.rstrip('\n')
            if line=='':
                continue
            if len(t)==0:
                continue
            if line.startswith('- EOF'):
                break
                
            if line.startswith('* Objvalue'):
                objvalue=4
                continue
            if line.startswith('- Variables'):
                partparsing='Variables'
                continue 
            
            if line.startswith('- Constraints'):
                partparsing='Constraints'
                continue         
            
            if partparsing=='Variables' :
                if line.startswith('V Name'): 
                    name=line.split(': ')[1].ljust(8)
                    var=variables[name]
                    continue  
     
                if line.startswith('V Value'): 
                    val1=float(line.split(':')[1].split('=')[0])
                    frac=line.split(':')[1].split('=')[1].split('/')
                    if len(frac)==1:
                        val=float(frac[0])
                    else:
                        val=float(frac[0])/float(frac[1])
                    if np.isnan(val):# happends with PEROLD
                        var['sol']=val1
                    else:
                        var['sol']=val
                    continue 
                
                if line.startswith('V State    : on lower'):      
                    var['sol']=var['LO']
                    continue  
                
                if line.startswith('V State    : on upper'):      
                    var['sol']=var['UP']
                    continue  
                
                if line.startswith('V State    : on both'): 
                    assert(var['UP']==var['LO'])
                    var['sol']=var['UP']
                    continue                  
                
                
                
                
        solution=np.array([vidtovar[i]['sol'] for i in range(nb_var)])
              
        r['solution']=solution
          
    return r 
           
            
            
def getProblem(pbname):
    thisfilepath=os.path.dirname(os.path.abspath(__file__))
    
    
    filenameLP  =os.path.join(thisfilepath,'data','netlib',pbname.upper()+'.SIF')
    filenameSol =os.path.join(thisfilepath,'data','perPlex',pbname.lower()+'.txt')
    
    if not os.path.isfile(filenameLP):
        urllib.urlretrieve ("ftp://ftp.numerical.rl.ac.uk/pub/cuter/netlib/%s.SIF"%pbname.upper(), filenameLP)	
    if not os.path.isfile(filenameSol):
        urllib.urlretrieve ("http://www.zib.de/koch/perplex/data/netlib/txt/%s.txt.gz"%pbname.lower(),filenameSol+'.gz')
        fgz=gzip.open(filenameSol+'.gz')
        f=open(filenameSol, 'w')
        for l in fgz.readlines():
            f.write(l)
        f.close()
    
    
    
    # netlib problems ftp://ftp.numerical.rl.ac.uk/pub/cuter/netlib.tar.gz
    # netlib exact solutions http://www.zib.de/koch/perplex/data/netlib/txt/

    fLP=open(filenameLP, 'r')
    if not filenameSol is None:
        fSol=open(filenameSol, 'r')
    else:
        fSol=None
        
    
    LPDict=MPSParser(fLP,fSol)
    return   LPDict     
           
    
    
    

if __name__ == "__main__":
    
    
   
    
    filenameLP ='./data/netlib/AFIRO.SIF'
    filenameSol='./data/perPlex/afiro.txt'
    fLP=open(filenameLP, 'r') 
    fsol=open(filenameSol, 'r') 
    LP=MPSParser(fLP,fsol)

         
        
 
    
