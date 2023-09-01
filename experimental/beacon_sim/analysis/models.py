"""
Module modeling simple systems for use with the KF
"""
import numpy as np

class SingleIntegrator():
    """
    Model for single integrator robot
    The states include x,y position
    The controls include x,y velocity
    The measurements include x,y position
    """
    def __init__(self, dt=1, Q=None, R=None, sigma_q=0.01, sigma_r=0.1):
        """
        Args
            dt time step
            Q process noise covariance matrix, x and y velocity
            R measurement noise covariance matrix, x and y position
        """
        self.dt=dt
        self.sigma_q=sigma_q
        self.sigma_r=sigma_r

        if Q is not None:
            self.Q=Q
        else:
            self.Q=self.sigma_q*np.identity(2)

        if R is not None:
            self.R=R
        else:
            self.R=self.sigma_r*np.identity(2)

        self.n_x=self.F().shape[0]
        self.n_u=self.L().shape[0]
        self.n_z=self.H().shape[0]

    def F(self):
        #jacobian of process wrt to states
        return np.array([[1.0, 0.1],[0.0, 1.0]])

    def L(self):
        #jacobian of process wrt to process noise
        return self.dt*np.identity(2)

    def H(self):
        #jacobian of observation wrt to states
        return np.identity(2)

    def M(self):
        #jacobian of observation wrt to observation noise
        return np.identity(2)

class DoubleIntegrator():
    """
    Model for double integrator robot.
    The states include x,y position and x,y velocity
    The controls include x,y accelerations
    The measurements include x,y position
    """
    def __init__(self, dt=1, Q=None, R=None, sigma_q=0.01, sigma_r=0.1):
        """
        Args
            dt time step
            Q process noise covariance matrix, x and y acceleration
            R measurement noise covariance matrix, x and y position
        """
        self.dt=dt
        self.sigma_q=sigma_q
        self.sigma_r=sigma_r

        if Q is not None:
            self.Q=Q
        else:
            self.Q=self.sigma_q*np.identity(2)

        if R is not None:
            self.R=R
        else:
            self.R=self.sigma_r*np.identity(2)

        self.n_x=self.F().shape[0]
        self.n_u=self.L().shape[0]
        self.n_z=self.H().shape[0]

    def F(self):
        F = np.identity(4)
        F[0,2] = self.dt
        F[1,3] = self.dt
        return F

    def L(self):
        L = np.zeros((4,2))
        L[2,0]=self.dt
        L[3,1]=self.dt
        return L

    def H(self):
        H = np.zeros((2,4))
        H[0,0]=1
        H[1,1]=1
        return H

    def M(self):
        return np.identity(2)
