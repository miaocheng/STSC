# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# stsc.py
# 
# This python file contains the definition of self-tuning spectral clustering method.
# Note: It is a trivial implementation, and has been put on hold for a term. Thus, either 
# accuracy and completeness are still suspended.
# 
# Miao Cheng 
# Email: miao_cheng@outlook.com
# Date: 2021-07
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
import numpy as np
from numpy import linalg as la

from functools import reduce
from scipy.optimize import minimize

from cala import *


class stsc(object):
    def __init__(self, X, kwargs):
        self.__X = X
        self.__xDim, self.__xSam = np.shape(X)
        
        if 'k' not in kwargs:
            kwargs['k'] = 5
            
        if 'c' not in kwargs:
            kwargs['c'] = 3
            
        if 't' not in kwargs:
            kwargs['t'] = 1
            
        if 'ctype' not in kwargs:
            kwargs['ctype'] = 'stsc'
            
        if 'atype' not in kwargs:
            kwargs['atype'] = 'self'
            
        if 'nIter' not in kwargs:
            kwargs['nIter'] = 1000
            
        if 'tol' not in kwargs:
            kwargs['tol'] = 1e-6
            
        # +++++ Parameters of STSC +++++
        if 'min_Cls' not in kwargs:
            kwargs['min_Cls'] = 2
            
        if 'max_Cls' not in kwargs:
            kwargs['max_Cls'] = 2
            
        self.__k = kwargs['k']
        self.__c = kwargs['c']
        self.__t = kwargs['t']
        self.__ctype = kwargs['stsc']
        self.__atype = kwargs['self']
        
        self.__nIter = kwargs['nIter']
        self.__tol = kwargs['tol']
        
        # +++++ Parameters of STSC +++++
        self.__min_Cls = kwargs['min_Cls']
        self.__max_Cls = kwargs['max_Cls']
        
        # ++++++++++ Initialization ++++++++++
        self.__getS()
        self.__getL()
        self.__normL()
        
        self.__cls = np.zeros((self.__c, self.__xSam))
        
        pass
    
    
    def __getS(self):
        D = eudist(self.__X, self.__X, False)
        
        if self.__atype == 'one':
            tmp = - D / self.__t
            
        elif self.__atype == 'self':
            M, index = sortMat(D, 'Row', 'Ascend')
            d = M[:, self.__k]
            dd = np.dot(d, np.transpose(d))
            tmp = - D / dd
            
        # ++++++++++ Exp Affinity ++++++++++
        S = np.exp(tmp)
        
        for i in range(self.__xSam):
            S[i, i] = 0
        
        N, index = sortMat(D, 'Row', 'Descend')
        ind = index[:, 0:self.__k]
        
        T = np.zeros((self.__xSam, self.__xSam))
        for i in range(self.__xSam):
            for j in range(self.__k):
                tid = ind[i, j]
                T[i, tid] = S[i, tid]
                
        T = T + T
        T = T * 0.5
        self.__S = T
        
        return True
        
        
    def __getL(self):
        tmp = np.sum(self.__S, axis=1)
        self.__D = np.diag(tmp)
        self.__L = self.__D - self.__S
        
        return True
    
    
    def __normL(self):
        d = np.diag(self.__D)
        d = d ** (- 0.5)
        dd = np.diag(d)
        tmp = np.dot(dd, self.__S)
        tmq = np.dot(tmp, dd)
        
        self.__nL = tmq
        
        return True
    
    
    def __updMeans(self):
        for i in range(self.__c):
            tmp = self.__cls[i, :]
            n = np.sum(tmp)
            tmq = repVec(tmp, self.__xDim)
            tmx = tmq * self.__X
            mx = np.sum(tmx, axis=1)
            mx = mx / n
            
            self.__km[:, i] = mx
            
        return True
    
    
    def __kmeans(self, X):
        xDim, xSam = np.shape(X)
        assert xDim == self.__c, 'The length of feature sizes are not identical !'
        
        # ++++++++++ Initialize the means ++++++++++
        ind = np.arange(xSam)
        np.random.shuffle(ind)
        ind = ind[0:self.__c]
        self.__km = X[:, ind]
        
        old_cls = self.__cls
        for ii in range(self.__nIter):
            d = eudist(X, self.__km, False)
            dd, index = sortMat(d, 'Row', 'Ascend')
            ind = index[:, 0]
            
            # ++++++++++ Aligned samples ++++++++++
            self.__cls = np.zeros((self.__c, self.__xSam))
            for i in range(xSam):
                tid = ind[i]
                self.__cls[tid, i] = 1
                
            self.__updMeans()
            
            # ++++++++++ Check the convergency ++++++++++
            tmp = self.__cls - old_cls
            tmq = tmp * tmp
            Obj = norm(tmq, 1)
            str_ = 'The %d' %ii + '-th iteration: %f' %Obj
            print(str_)
            
            if Obj < self.__tol:
                break
            
            old_cls = self.__cls
            
        return Obj
        
        
    def __njw(self):
        U, s, V = la.svd(self.__nL, full_matrices=False)
        V = np.transpose(V)
        s, r = getRank(s)
        
        # ++++++++++ Normalization ++++++++++
        U = U[:, 0:r]
        cc = U[:, 0:self.__c]
        tmp = cc * cc
        tmq = sum(tmp, axis=1)
        tmq = np.sqrt(tmq)
        tm = tmp / tmq
        tm = np.transpose(tm)
        
        self.__kmeans(tm)
        
        return True
    
    
    # ++++++++++ Self-tuning clustering ++++++++++
    def __GivensRotation(self, i, j, theta, size):
        g = np.eye(size)
        c = np.cos(theta)
        s = np.sin(theta)
        g[i, i] = 0
        g[j, j] = 0
        g[j, i] = 0
        g[i, j] = 0
        ii_mat = np.zeros_like(g)
        ii_mat[i, i] = 1
        jj_mat = np.zeros_like(g)
        jj_mat[j, j] = 1
        ji_mat = np.zeros_like(g)
        ji_mat[j, i] = 1
        ij_mat = np.zeros_like(g)
        ij_mat[i, j] = 1
        
        return g + c * ii_mat + c * jj_mat + s * ji_mat - s * ij_mat    
    
    
    def __generate_list(self, ij_list, theta_list, size):
        return [self.__GivensRotation(ij[0], ij[1], theta, size)
                for ij, theta in zip(ij_list, theta_list)]
    
    
    def __rotation(self, X, c):
        ij_list = [(i, j) for i in range(c) for j in range(c) if i < j]
        
        def cost(self, X, c, ij_list, theta_list):
            U_list = self.__generate_list(ij_list, theta_list, c)
            R = reduce(np.dot, U_list, np.eye(c))
            Z = X.dot(R)
            M = np.max(Z, axis=1, keepdims=True)
            N = np.sum((Z / M) ** 2)    
            
            return N        
        
        theta_list_init = np.array([0.0] * int(c * (c - 1) / 2))
        opt = minimize(cost,
                       x0 = theta_list_init,
                       method = 'CG',
                       jac = grad(cost),
                       options = {'disp': False})
        
        return opt.fun, reduce(np.dot, self.__generate_list(ij_list, opt.x, c), np.eye(c))
    
    
    def __reformat(labels, n):
        zipped_data = zip(labels, range(n))
        zipped_data = sorted(zipped_data, key=lambda x: x[0])
        grouped_feature_id = [[j[1] for j in i[1]] for i in groupby(zipped_data, lambda x: x[0])]      
        
        return grouped_feature_id        
        
    
    def __stsc(self):
        U, s, V = la.svd(self.__nL, full_matrices=False)
        V = np.transpose(V)
        s, r = getRank(s)
        
        #t = revArr(s)
        ss = np.sum(s)
        if ss < 2:
            self.__max_Cls = 2
        else:
            self.__max_Cls = int(ss)
            
        re = []
        for i in range(self.__min_Cls, self.__max_Cls + 1):
            tmv = U[:, :i]
            cost, tmr = self.__rotation(tmv, i)
            re.append((cost, tmv.dot(tmr)))
            
            str_ = 'n_cluster: %d' %c + '\t cost: %f' %cost
            print(str_)
            
        COST, Z = sorted(re, key = lambda x: x[0])[0]
        tm = self.__reformat(np.argmax(Z, axis=1), Z.shape[0])
        
        return tm
    
    
    def Learn(self):
        if self.__ctype == 'stsc':
            self.__stsc()
            
        elif self.__ctype == 'njw':
            self.__njw()
            
        return True
    
    
    def getLabel(self):
        B, index = iMax(self.__cls, axis=0)
        labels = index
        
        return labels
    
    
    
            
            
        
        
        
