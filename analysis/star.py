"""
Utilities for star product
"""
import numpy as np
from numpy.linalg import inv
# import itertools

def sblocks(s):
    """
    Return blocks of 2n x 2n matrix s = [[a,b],[c,d]]
    """
    n = s.shape[0]//2
    a=s[0:n,0:n]
    b=s[0:n,n:2*n]
    c=s[n:2*n,0:n]
    d=s[n:2*n,n:2*n]
    return a, b, c, d

def sprod(s0,s1):
    """
    Computes Redheffer star product s=s0*s1 of 2nx2n matrices s0 and s1
    """
    a,b,c,d = sblocks(s0)
    w,x,y,z = sblocks(s1)

    i=np.identity(a.shape[0])
    blk0 = w @ inv(i-b@y) @ a
    blk1 = x + w @ inv(i-b@y) @ b @ z
    blk2 = c + d @ inv(i-y@b) @ y @ a
    blk3 = d @ inv(i-y@b) @ z

    return np.block([[blk0, blk1],[blk2,blk3]])

def scattering_from_transfer(a):
    a,b,c,d = sblocks(a)
    d_inv = inv(d)
    blk0 = a - b @ d_inv @ c
    blk1 = b @ d_inv
    blk2 = -d_inv @ c
    blk3 = d_inv
    return np.block([[blk0, blk1],[blk2,blk3]])

def transfer_from_scattering(a):
    return scattering_from_transfer(a);
