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

    def predict_covariance(self,P,backward=False):
        if not backward:
            P = self.model.F() @ P @ self.model.F().transpose() + self.model.L() @ self.model.Q @ self.model.L().transpose()
        else:
            P = inv(self.model.F()) @ (P - self.model.L() @ self.model.Q @ self.model.L().transpose()) @ inv(self.model.F().transpose())
        return P

    def update_covariance(self,P,backward=False):
        if not backward:
            n = P.shape[0]
            S = self.model.H() @ P @ self.model.H().transpose() + self.model.M() @ self.model.R @ self.model.M().transpose()
            K= P @ self.model.H().transpose() @ inv(S)
            return (np.identity(n) - K @ self.model.H()) @ P
        else:
            n = P.shape[0]
            Sbar = self.model.M() @ self.model.R @ self.model.M().transpose() - self.model.H() @ P @ self.model.H().transpose()
            Kbar = P @ self.model.H().transpose() @ inv(Sbar)
            return (np.identity(n) + Kbar @ self.model.H()) @ P

    def predict_information(self, info, backward=False):
        return inv(self.predict_covariance(inv(info), backward=backward))

    def update_information(self, info, backward=False):
        meas_info = self.model.H().transpose() @ inv(self.model.R) @ self.model.H()
        if not backward:
            return info + meas_info
        else:
            return info - meas_info


    def predict(self, state, backward=False):
        if self.filter_type == FilterType.COVARIANCE:
            return self.predict_covariance(state, backward=backward)
        else:
            return self.predict_information(state, backward=backward)

    def update(self, state, backward=False):
        if self.filter_type == FilterType.COVARIANCE:
            return self.update_covariance(state, backward=backward)
        else:
            return self.update_information(state, backward=backward)



    #functions for propagating belief
    def prop(self, P, k=1, backward=False):
        """
        Propagates a belief P through k steps using EKF equations
        """
        if not backward:
            for i in range(k):
                P = self.predict(P, backward=False)
                P = self.update(P, backward=False)
        else:
            #combined equation for backward prop
            #S_bar = R - HPH'
            #K_bar = P * H' * S_bat^{-1}
            #P0 = F^{-1} * [ (I + K_bar*H)*P - Q ] * F'^{-1}'
            for i in range(k):
                P = self.update(P, backward=True)
                P = self.predict(P, backward=True)
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
        F = self.model.F()
        Q = self.model.Q
        inv_F = inv(F)
        if self.filter_type == FilterType.COVARIANCE:
            block_1 = F
            block_2 = Q @ inv_F.transpose()
            block_3 = np.zeros_like(block_1)
            block_4 = inv_F.transpose()
        else:
            block_1 = inv_F.transpose()
            block_2 = np.zeros_like(block_1)
            block_3 = Q @ inv_F.transpose()
            block_4 = F

        return np.block([[block_1, block_2], [block_3, block_4]])

    def compute_one_step_measurement_tf(self):
        M = self.model.M()
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
        F = self.model.F()
        Q = self.model.Q
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
        M = self.model.F()
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
