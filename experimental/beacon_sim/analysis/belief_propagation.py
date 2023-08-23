"""
Module modeling implementing belief progration for EKF with one-step transfor functions
"""
import numpy as np
from numpy.linalg import inv, eig, cond, det, norm

from star import sblocks, sprod

class BeliefPropagation:
    """
    Implements belief propagation using first-order approximation of belief dynamics
    """
    def __init__(self, model):
        self.model=model

    def remap_vars(self):
        G = self.model.F()
        R = self.model.L() @ self.model.Q @ self.model.L().transpose()
        M = self.model.H().transpose() @ inv(self.model.R) @ self.model.H()
        return G, R, M

    def predict(self,P):
        P = self.model.F() @ P @ self.model.F().transpose() + self.model.L() @ self.model.Q @ self.model.L().transpose()
        return P

    def update(self,P):
        n = P.shape[0]
        S = self.model.H() @ P @ self.model.H().transpose() + self.model.M() @ self.model.R @ self.model.M().transpose()
        K= P @ self.model.H().transpose() @ inv(S)
        return (np.identity(n) - K @ self.model.H()) @ P

    #functions for propagating belief
    def prop(self, P, k=1):
        """
        Propagates covariance matrix P through k steps using EKF equations
        """
        for i in range(k):
            P = self.predict(P)
            P = self.update(P)
        return P

    def prop_tf(self, P, k=1, only_predict=False, return_tf=False):
        """
        Propagates covariance matrix P through k steps using one-step transfer function
        """
        gamma = self.compute_tf(k=k, only_predict=only_predict)
        n = self.model.n_x
        I = np.identity(n)
        if return_tf:
            return gamma
        else:
            psi = np.block([[P],[I]])
            gamma_p = gamma @ psi
            B = gamma_p[0:n,0:n]
            C = gamma_p[n:2*n,0:n]
            return B @ inv(C)

    def prop_scatter(self, P, k=1, only_predict=False, return_scatter=False):
        """
        Propagates covariance matrix P through k steps using scattering matrices
        """
        s = self.compose_scatter_from_cov(P)
        if k>0:
            s = sprod(s, self.compute_scatter(k=k, only_predict=only_predict))
        n = s.shape[0]//2
        if return_scatter:
            return s
        else:
            return s[0:n,n:2*n]

    #utilities for scattering form
    def compose_scatter(self, only_predict=False):
        n=self.model.n_x
        I = np.identity(n)
        Z = np.zeros((n,n))
        G, R, M = self.remap_vars()

        sc = np.block([[G, R],[Z,G.transpose()]]) #control step
        sm = np.block([[I, Z],[-M,I]]) #update step

        if only_predict:
            return sc
        else:
            return sprod(sc,sm)

    def compose_scatter_from_cov(self, P):
        n = P.shape[0]
        I = np.identity(n)
        Z = np.zeros((n,n))
        return np.block([[I, P],[Z,I]])

    def compute_scatter(self, k=1, only_predict=False):
        s = self.compose_scatter(only_predict=only_predict)
        for i in range(k-1):
            s = sprod(s, self.compose_scatter(only_predict=only_predict))
        return s

    def apply_scatter_to_cov(self, P, s):
        s0= self.compose_scatter_from_cov(P)
        s = sprod(s0, s)
        n = s.shape[0]//2
        return s[0:n,n:2*n]

    #utilities for tf form
    def compute_tf(self, k=1, only_predict=False):
        n = self.model.n_x
        I = np.identity(n)
        Z = np.zeros((n,n))
        G, R, M = self.remap_vars()

        gamma = np.identity(2*n)
        for i in range(k):
            if only_predict:
                X = np.block([[Z, I],[I,Z]])
                Y = np.block([[Z, inv(G.transpose())],[G,R@inv(G.transpose())]])
            else:
                X = np.block([[Z, I],[I,M]])
                Y = np.block([[Z, inv(G.transpose())],[G,R@inv(G.transpose())]])
            gamma = X @ Y @ gamma
        return gamma

    def apply_tf_to_cov(self, P, gamma):
        n = self.model.n_x
        I = np.identity(n)
        psi = np.block([[P],[I]])
        gamma_p = gamma @ psi
        B = gamma_p[0:n,0:n]
        C = gamma_p[n:2*n,0:n]
        return B @ inv(C)
