from .Dynamics import DiscreteTimeDynamics
import numpy as np
from scipy.linalg import solve_discrete_are
from nn_closed_loop.utils.mpc import control_mpc


class DoubleIntegrator(DiscreteTimeDynamics):
    def __init__(self):

        self.continuous_time = False

        At = np.array([[1, 1], [0, 1]])
        bt = np.array([[0.5], [1]])
        ct = np.array([0.0, 0.0]).T

        # u_limits = None
        u_limits = np.array([[-1.0, 1.0]])  # (u0_min, u0_max)

        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits)

        self.cmap_name = "tab10"

    def control_mpc(self, x0):
        # LQR-MPC parameters
        if not hasattr(self, "Q"):
            self.Q = np.eye(2)
        if not hasattr(self, "R"):
            self.R = 1
        if not hasattr(self, "Pinf"):
            self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

        return control_mpc(
            x0,
            self.At,
            self.bt,
            self.ct,
            self.Q,
            self.R,
            self.Pinf,
            self.u_limits[:, 0],
            self.u_limits[:, 1],
            n_mpc=10,
            debug=False,
        )
