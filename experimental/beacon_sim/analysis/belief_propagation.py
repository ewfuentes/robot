"""
Module modeling implementing belief progration for EKF with one-step transfor functions
"""
import numpy as np
from numpy.linalg import inv, eig, cond, det, norm
import enum

from experimental.beacon_sim.analysis.star import sblocks, sprod

class FilterType(enum.Enum):
    COVARIANCE = enum.auto()
    INFORMATION = enum.auto()

class BeliefPropagation:
    """
    Implements belief propagation using first-order approximation of belief dynamics
    """
    def __init__(self, model, filter_type=FilterType.COVARIANCE):
        self.model = model
        self.filter_type = filter_type

    def remap_vars(self):
        G = self.model.F()
        R = self.model.L() @ self.model.Q @ self.model.L().transpose()
        M = self.model.H().transpose() @ inv(self.model.R) @ self.model.H()
        return G, R, M

    def predict_covariance(self,P):
        P = self.model.F() @ P @ self.model.F().transpose() + self.model.L() @ self.model.Q @ self.model.L().transpose()
        return P

    def update_covariance(self,P):
        n = P.shape[0]
        S = self.model.H() @ P @ self.model.H().transpose() + self.model.M() @ self.model.R @ self.model.M().transpose()
        K= P @ self.model.H().transpose() @ inv(S)
        return (np.identity(n) - K @ self.model.H()) @ P

    def predict_information(self, info):
        return inv(self.predict_covariance(inv(info)))

    def update_information(self, info):
        meas_info = self.model.H().transpose() @ inv(self.model.R) @ self.model.H()
        return info + meas_info


    def predict(self, state):
        if self.filter_type == FilterType.Covariance:
            return self.predict_covariance(state)
        else:
            return self.predict_information(state)

    def update(self, state):
        if self.filter_type == FilterType.Covariance:
            return self.update_covariance(state)
        else:
            return self.update_information(state)



    #functions for propagating belief
    def prop(self, P, k=1):
        """
        Propagates a belief P through k steps using EKF equations
        """
        for i in range(k):
            P = self.predict(P)
            P = self.update(P)
        return P

    def prop_tf(self, P, k=1, only_predict=False, return_tf=False):
        """
        Propagates a belief P through k steps using one-step transfer function
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
        Propagates a belief P through k steps using scattering matrices
        """
        s = self.compose_scatter_from_cov(P)
        if k>0:
            s = sprod(s, self.compute_scatter(k=k, only_predict=only_predict))
        n = s.shape[0]//2
        if return_scatter:
            return s
        else:
            return s[0:n,n:2*n]

    def compute_one_step_process_tf(self):
        if self.filter_type == FilterType.COVARIANCE:
            block_1 = self.models.F()
            block_2 = self.models.Q @ inv(self.model.F().transpose())
            block_3 = np.zeros_like(block_1)
            block_4 = inv(block_1.transpose())
        else:
            block_1 = inv(self.models.F().transpose())
            block_2 = np.zeros_like(block_1)
            block_3 = self.models.Q @ inv(self.model.F().transpose())
            block_4 = self.models.F()

        return np.block([[block_1, block_2], [block_3, block_4]])

    def compute_one_step_measurement_tf(self):
        M = self.models.M()
        I = np.eye(M.shape[0])
        if self.filter_type == FilterType.COVARIANCE:
            block_1 = I
            block_2 = np.zeros_like(M)
            block_3 = M
            block_4 = I
        else:
            block_1 = I
            block_2 = M
            block_3 = np.zeros_like(M)
            block_4 = I
        return np.block([[block_1, block_2], [block_3, block_4]])

    def compute_one_step_process_scatter(self):
        F = self.models.F()
        Q = self.models.Q
        if self.filter_type == FilterType.COVARIANCE:
            block_1 = F
            block_2 = Q
            block_3 = np.zeros_like(Q)
            block_4 = F.transpose()
        else:
            inv_F = inv(F)
            block_1 = inv_F.transpose()
            block_2 = np.zeros_like(Q)
            block_3 = -inv_F @ Q @ inv_F.transpose()
            block_4 = inv_F

        return np.block([[block_1, block_2], [block_3, block_4]])

    def compute_one_step_measurement_scatter(self):
        M = self.models.F()
        I = np.identity(M.shape[0])
        if self.filter_type == FilterType.COVARIANCE:
            block_1 = I
            block_2 = np.zeros_like(M)
            block_3 = -M
            block_4 = I
        else:
            block_1 = I
            block_2 = M
            block_3 = np.zeros_like(M)
            block_4 = I

        return np.block([[block_1, block_2], [block_3, block_4]])

    #utilities for scattering form
    def compose_scatter(self, only_predict=False):
        sc = self.compute_one_step_process_scatter()
        sm = self.compute_one_step_measurement_scatter()

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

        tf_process = self.compute_one_step_process_tf()
        tf_meas = self.compute_one_step_measurement_tf()

        gamma = np.identity(2*n)
        for i in range(k):
            if only_predict:
                X = tf_process
            else:
                X = tf_meas @ tf_process
            gamma = X @ gamma
        return gamma

    def apply_tf_to_cov(self, P, gamma):
        n = self.model.n_x
        I = np.identity(n)
        psi = np.block([[P],[I]])
        gamma_p = gamma @ psi
        B = gamma_p[0:n,0:n]
        C = gamma_p[n:2*n,0:n]
        return B @ inv(C)
