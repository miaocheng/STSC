
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# tolo.py
# This python file contains the essential definition of calculation functions.
# Coded by Miao Cheng
# Date: 2020-10-15
# All rights reserved
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import random
from numpy import linalg as la

from cala import *


def toTxt(fname, str_):
    f = open(fname, 'a')
    f.write(str_)

    return True

def vMax(u, v):
    m = len(u)
    n = len(v)
    assert m == n, 'The length of u and v are not identical !'
    
    w = []
    for i in range(m):
        tmp = u[i]
        tmq = v[i]
        
        tmr = max(tmp, tmq)
        w.append(tmr)
    
    return w

def vMin(u, v):
    m = len(u)
    n = len(v)
    assert m == n, 'The length of u and v are not identical !'
    
    w = []
    for i in range(m):
        tmp = u[i]
        tmq = v[i]
        
        tmr = min(tmp, tmq)
        w.append(tmr)
        
    return w

def rowNorm(M):
    nRow, nCol = np.shape(M)
    
    tmp = M * M
    d = np.sum(tmp, axis=1)
    d = np.sqrt(d)
    tmp = np.tile(d, (nCol, 1))
    tmp = np.transpose(tmp)
    M = M / tmp    
    
    return M


def ifSingular(M):
    nRow, nCol = np.shape(M)
    
    for i in range(nRow):
        for j in range(nCol):
            if abs(M[i, j]) < 1e-6:
                M[i, j] = 1
                
    return M


def abZ(M):
    nRow, nCol = np.shape(M)
    
    M = justNorm(M)
    
    tmp = np.min(M)
    M = M - tmp
    tmq = np.max(M)
    M = M / tmq
    
    return M
    

def toNorm(W):
    nLen = len(W)
    for i in range(nLen):
        tmw = W[i]
        tmw = justNorm(tmw)
        W[i] = tmw
        
    return W


def toOne(M):
    nRow, nCol = np.shape(M)
    
    for i in range(nRow):
        for j in range(nCol):
            if M[i, j] < 0:
                M[i, j] = 1
                
    return M


def onlyPos(M):
    nRow, nCol = np.shape(M)
    
    for i in range(nRow):
        for j in range(nCol):
            if M[i, j] < 0:
                M[i, j] = 0
                
    return M
    

def noInf(M):
    nRow, nCol = np.shape(M)
    idx = []
    idy = []
    
    for i in range(nRow):
        for j in range(nCol):
            if np.isinf(M[i, j]):
                M[i, j] = 0
                idx.append(i)
                idy.append(j)
                
    return M, idx, idy
    
    
def onediv(M):
    nRow, nCol = np.shape(M)
    
    for i in range(nRow):
        for j in range(nCol):
            if np.isinf(M[i, j]):
                M[i, j] = 0
            else:
                M[i, j] = float(1) / M[i, j]
                
    return M


def getLabel(L):
    nCls, nSam = np.shape(L)
    
    M, index = iMax(L, axis=0)
    
    return index


def getAccuracy(L, Labels):
    n = len(L)
    
    acc = 0
    for i in range(n):
        if L[i] == Labels[i]:
            acc = acc + 1
            
    accuracy = float(acc) / n
    
    return accuracy, acc


def getAA(X, Y):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    assert xSam == ySam, 'The sample amounts of X and Y are not identical !'
    
    #tmp = np.dot(X, np.transpose(X))
    #tmp = np.sqrt(np.trace(tmp))
    #tmq = np.dot(Y, np.transpose(Y))
    #tmq = np.sqrt(np.trace(tmq))
    #tm = tmp * tmq
    
    #tn = np.dot(X, np.transpose(Y))
    #tn = np.trace(tn)
    
    tmp = np.dot(np.transpose(X), X)
    tm = np.dot(tmp, tmp)
    tm = np.sqrt(np.trace(tm))
    
    tmq = np.dot(np.transpose(Y), Y)
    tn = np.dot(tmq, tmq)
    tn = np.sqrt(np.trace(tn))
    
    tmn = np.dot(tmp, tmq)
    tmn = np.abs(tmn)
    tmn = np.trace(tmn)
    
    cc = tmn / (tm * tn)
    
    return cc
    

def getCC(X, Y):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    #assert xDim == yDim, 'The dimensionalities of X and Y are not identical !'
    assert xSam == ySam, 'The sample amounts of X and Y are not identical !'
    
    #aa = getAA(X, Y)
    
    U, s, V = la.svd(X)
    s, m = getRank(s)
    U = U[:, 0:m]
    ss = s ** (-1)
    ss = np.diag(ss)
    tmp = np.dot(U, ss)
    tmx = np.dot(np.transpose(tmp), X)
    
    U, s, V = la.svd(Y)
    s, m = getRank(s)
    U = U[:, 0:m]
    ss = s ** (-1)
    ss = np.diag(ss)
    tmq = np.dot(U, ss)
    tmy = np.dot(np.transpose(tmq), Y)
    
    tm = np.dot(tmx, np.transpose(tmy))
    U, s, V = la.svd(tm)
    s, m = getRank(s)
    U = U[:, 0:m]
    V = V[:, 0:m]
    
    tmp = np.dot(tmp, U)
    tmq = np.dot(tmq, V)
    
    aa = np.sum(s)
    
    return tmp, tmq, aa


def getACC(X, Y, k):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    assert xDim == yDim, 'The dimensionalities of X and Y are not identical !'
    
    if xSam <= ySam:
        S = X
        T = Y
    elif xSam > ySam:
        T = X
        S = Y
        
    aa = 0
    cc = 0
    for i in range(k):
        sDim, sSam = np.shape(S)
        tDim, tSam = np.shape(T)
        
        ind = list(range(tSam))
        random.shuffle(ind)
        ind = ind[0:sSam]
        
        tmt = T[:, ind]
        tmp, tmq, ta = getCC(S, tmt)
        
        aa = aa + ta
        
        S = np.dot(np.transpose(tmp), S)
        T = np.dot(np.transpose(tmq), tmt)
        
        tc = getAA(S, T)
        cc = cc + tc
        
    aa = aa / k
    cc = cc / k
    
    return aa, cc
    
    
def sqInv(M):
    mRow, mCol = np.shape(M)
    assert mRow == mCol, 'The length of dimensionalities are not identical !'
    
    if mRow <= 1000:
        U, s, V = la.svd(M)
        s, r = getRank(s)
        U = U[:, 0:r]
        V = V[:, 0:r]
        ss = s ** (-1)
        dd = np.diag(ss)
        
        tmp = np.dot(V, dd)
        N = np.dot(tmp, np.transpose(U))
        
    elif mRow > 1000:
        index = list(np.arange(mCol))
        np.random.shuffle(index)
        ind = index[0:500]
        
        tm = M[:, ind]
        tmp = np.dot(np.transpose(tm), M)
        tn = np.dot(tmp, tm)
        
        U, s, V = la.svd(tn)
        s, r = getRank(s)
        U = U[:, 0:r]
        V = V[:, 0:r]
        ss = s ** (0.5)
        dd = np.diag(ss)
        
        tmp = np.dot(tm, U)
        tmq = np.dot(tmp, dd)
        U, s, V = la.svd(tmq)
        s, r = getRank(s)
        U = U[:, 0:r]
        ss = ss ** 2
        ss = ss ** (-1)
        dd = np.diag(ss)
        
        tmp = np.dot(U, dd)
        N = np.dot(tmp, np.transpose(U))
        
    return N
    
    
def mySVD(M):
    mRow, mCol = np.shape(M)
    
    if mRow > 1000:
        index = list(np.arange(mCol))
        np.random.shuffle(index)
        ind = index[0:500]
        
        tm = M[:, ind]
        tmp = np.dot(np.transpose(tm), M)
        tn = np.dot(tmp, np.transpose(tmp))
        U, s, V = la.svd(tn)
        s, r = getRank(s)
        tU = U[:, 0:r]
        
        s = s ** (0.5)
        d = np.diag(s)
        tmp = np.dot(tm, tU)
        tmq = np.dot(tmp, d)
        U, s, V = la.svd(tmq)
        
        tm = np.dot(np.transpose(U), M)
        U, s, V = la.svd(tm)
        s, r = getRank(s)
        U = U[:, 0:r]
        V = V[:, 0:r]
        
        U = np.dot(tU, U)
        
    elif mCol > 1000:
        N = np.transpose(M)
        mRow, mCol = np.shape(N)
        
        index = list(np.arange(mCol))
        np.random.shuffle(index)
        ind = index[0:500]
        
        tm = N[:, ind]
        tmp = np.dot(np.transpose(tm), N)
        tn = np.dot(tmp, np.transpose(tmp))
        U, s, V = la.svd(tn)
        s, r = getRank(s)
        tU = U[:, 0:r]
        
        s = s ** (0.5)
        d = np.diag(s)
        tmp = np.dot(tm, tU)
        tmq = np.dot(tmp, d)
        U, s, V = la.svd(tmq)
        
        tm = np.dot(np.transpose(U), N)
        U, s, V = la.svd(tm)
        s, r = getRank(s)
        U = U[:, 0:r]
        V = V[:, 0:r]
        
        U = np.dot(tU, U)
        
        tU = U
        U = V
        V = tU
        
    else:
        U, s, V = la.svd(M)
        s, r = getRank(s)
        U = U[:, 0:r]
        V = V[:, 0:r]
        
        
    return U, s, V


def checkNan(M):
    mRow, mCol = np.shape(M)
    
    for i in range(mRow):
        for j in range(mCol):
            tmp = M[i, j]
            if np.isnan(tmp):
                M[i, j] = 0
                
                
    return M
    
        
        
        
        
        
        
        
    
    
    