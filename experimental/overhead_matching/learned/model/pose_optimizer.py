
import common.torch.load_torch_deps
import torch
import numpy as np
import sympy
import itertools
import enum
import matplotlib.pyplot as plt

from sdprlayers.layers.sdprlayer import SDPRLayer


class OptimizationType(enum.Enum):
    POINT_CLOUD_REGISTRATION = enum.auto()  # 3D-3D correspondences
    POSE_ESTIMATION = enum.auto()  # 3D-2D correspondences


def compute_pnp_loss_coeffs():
    """
    Compute the loss coefficients for matching a bearing vectors to a point in space
    """
    # (x_a, y_a) is the point in the world frame
    # (c_b, s_b) represents the cosine and sin of the bearing angle
    x_a, y_a, c_b, s_b = sympy.symbols("x_a, y_a, c_b, s_b")
    # These represent the camera_from_world transformation
    s, c, t_x, t_y = sympy.symbols("s c t_x t_y")

    b_from_a_rot = sympy.Matrix([[c, -s], [s, c]])
    b_from_a_trans = sympy.Matrix([[t_x], [t_y]])
    p_in_a = sympy.Matrix([[x_a], [y_a]])

    u = sympy.Matrix([[c_b], [s_b]])

    proj_p_in_b = b_from_a_rot @ p_in_a + b_from_a_trans
    proj_p_in_b_along_u = (u.T @ proj_p_in_b)[0, 0] * u
    error = proj_p_in_b - proj_p_in_b_along_u
    loss = (error.T @ error)[0, 0]
    loss = sympy.expand(loss)
    loss_coeffs = loss.collect([c, s, c*s, t_x, t_y, c*t_x, c*t_y, s*t_x, s*t_y, t_x * t_y], evaluate=False)

    for k, v in loss_coeffs.items():
        print(k, v)

    loss_coeffs = {k: sympy.lambdify([x_a, y_a, c_b, s_b], v) for k, v in loss_coeffs.items()}
    return loss_coeffs


def compute_loss_coeffs():
    # NOTE: if changing this, double check seralization below
    x_a, y_a, x_b, y_b = sympy.symbols("x_a y_a x_b y_b")
    s, c, t_x, t_y = sympy.symbols("s c t_x t_y")
    b_from_a_rot = sympy.Matrix([[c, -s], [s, c]])
    b_from_a_trans = sympy.Matrix([[t_x], [t_y]])
    p_a = sympy.Matrix([[x_a], [y_a]])
    p_b = sympy.Matrix([[x_b], [y_b]])

    proj_p_in_b = b_from_a_rot @ p_a + b_from_a_trans
    error = p_b - proj_p_in_b
    loss = (error.T @ error)[0, 0]
    loss = sympy.expand(loss)
    loss_coeffs = loss.collect([c, s, t_x, t_y, c*t_x, c*t_y, s*t_x, s*t_y], evaluate=False)

    loss_coeffs = {k: sympy.lambdify([x_a, y_a, x_b, y_b], v) for k, v in loss_coeffs.items()}
    return loss_coeffs


def build_pnp_q_matrix(associations, pt_in_a, bearing_in_b, loss_coeffs):
    assert associations.shape[0] == pt_in_a.shape[0]
    assert associations.shape[0] == bearing_in_b.shape[0]
    assert associations.shape[1] >= pt_in_a.shape[1]
    assert associations.shape[2] >= bearing_in_b.shape[1]
    batch_size, num_obj_in_a, _ = pt_in_a.shape
    _, num_obj_in_b = bearing_in_b.shape

    uv_in_b = torch.empty_like(pt_in_a)
    uv_in_b[..., 0] = torch.cos(bearing_in_b)
    uv_in_b[..., 1] = torch.sin(bearing_in_b)

    # The pose optimization problem is equivalent to a polynomial minimization
    # problem. To formulate the polynomial minimization as an SDP, we need to
    # create a matrix of polynomial coefficents.

    # We build up the coefficient matrix by combining products of the elements in a and b
    pt_in_a = pt_in_a.unsqueeze(-2)
    uv_in_b = uv_in_b.unsqueeze(-3)
    x_a = pt_in_a[..., 0]
    y_a = pt_in_a[..., 1]
    c_b = uv_in_b[..., 0]
    s_b = uv_in_b[..., 1]

    t_x, t_y, c, s = sympy.symbols("t_x t_y c s")

    q_per_pair = torch.empty((batch_size, num_obj_in_a, num_obj_in_b, 5, 5))
    for (i, s_i), (j, s_j) in itertools.combinations_with_replacement(enumerate([1, t_x, t_y, c, s]), 2):
        term = s_i * s_j
        value = loss_coeffs.get(term, lambda **kwargs: 0)(x_a=x_a, y_a=y_a, c_b=c_b, s_b=s_b)
        if i == j:
            q_per_pair[..., i, j] = value
        else:
            q_per_pair[..., i, j] = 0.5 * value
            q_per_pair[..., j, i] = 0.5 * value

    # The output tensor is formed by multiplying elementwise with the
    # batch x num_obj_in_a x num_obj_in_b attention matrix and then summing out
    # that dimension
    # batch x num_obj_in_a x num_obj_in_b x 5 x 5
    associations = associations.unsqueeze(-1).unsqueeze(-1)

    weighted_q_per_pair = associations[:, :num_obj_in_a, :num_obj_in_b, ...] * q_per_pair

    q = torch.sum(weighted_q_per_pair, dim=(1, 2))

    # The output is a tensor of shape batch x 5 x 5 since we have 5 monomials
    return q


def build_q_matrix(associations, pt_in_a, pt_in_b, loss_coeffs):
    assert associations.shape[0] == pt_in_a.shape[0]
    assert associations.shape[0] == pt_in_b.shape[0]
    assert associations.shape[1] >= pt_in_a.shape[1]
    assert associations.shape[2] >= pt_in_b.shape[1]
    batch_size, num_obj_in_a, _ = pt_in_a.shape
    _, num_obj_in_b, _ = pt_in_b.shape

    # The pose optimization problem is equivalent to a polynomial minimization
    # problem. To formulate the polynomial minimization as an SDP, we need to
    # create a matrix of polynomial coefficents.

    # We build up the coefficient matrix by combining products of the elements in a and b
    pt_in_a = pt_in_a.unsqueeze(-2)
    pt_in_b = pt_in_b.unsqueeze(-3)
    x_a = pt_in_a[..., 0]
    y_a = pt_in_a[..., 1]
    x_b = pt_in_b[..., 0]
    y_b = pt_in_b[..., 1]

    t_x, t_y, c, s = sympy.symbols("t_x t_y c s")

    q_per_pair = torch.empty((batch_size, num_obj_in_a, num_obj_in_b, 5, 5))
    for (i, s_i), (j, s_j) in itertools.combinations_with_replacement(enumerate([1, t_x, t_y, c, s]), 2):
        term = s_i * s_j
        value = loss_coeffs.get(term, lambda **kwargs: 0)(x_a=x_a, y_a=y_a, x_b=x_b, y_b=y_b)
        if i == j:
            q_per_pair[..., i, j] = value
        else:
            q_per_pair[..., i, j] = 0.5 * value
            q_per_pair[..., j, i] = 0.5 * value

    # The output tensor is formed by multiplying elementwise with the
    # batch x num_obj_in_a x num_obj_in_b attention matrix and then summing out
    # that dimension
    # batch x num_obj_in_a x num_obj_in_b x 5 x 5

    associations = associations.unsqueeze(-1).unsqueeze(-1)

    weighted_q_per_pair = associations[:, :num_obj_in_a, :num_obj_in_b, ...] * q_per_pair

    q = torch.sum(weighted_q_per_pair, dim=(1, 2))

    # The output is a tensor of shape batch x 5 x 5 since we have 5 monomials
    return q


class PoseOptimizerLayer(torch.nn.Module):
    def __init__(self, optimization_type: OptimizationType):
        """
        We want to pose the polynomial optimization problem as a QCQP where the monomials are
        x = [1 t_x t_y cos(theta) sin(theta)]

        under the constraint that cos(theta)**2 + sin(theta)**2 = 1
        """
        super().__init__()

        self._optimization_type = optimization_type

        # NOTE: if changing this, double check seralization below
        self._optimizer = SDPRLayer(
            n_vars=5,
            constraints=[
                # sin**2 + cos**2 - 1 = 0
                np.array([
                    [-1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0]])
            ]
        )

        self._configure_for_optimization_type(self._optimization_type)

    def _configure_for_optimization_type(self, optimization_type: OptimizationType):
        match optimization_type:
            case OptimizationType.POINT_CLOUD_REGISTRATION:
                self._loss_coeffs = compute_loss_coeffs()
                self._build_q_matrix = build_q_matrix
            case OptimizationType.POSE_ESTIMATION:
                self._loss_coeffs = compute_pnp_loss_coeffs()
                self._build_q_matrix = build_pnp_q_matrix

    def __getstate__(self):
        return (
            self._optimizer.n_vars,
            self._optimizer.use_dual,
            self._optimizer.constr_list,
            self._optimization_type
        )

    def __setstate__(self, state):
        super().__init__()
        self._optimization_type = state[3]
        self._configure_for_optimization_type(self._optimization_type)

        self._optimizer = SDPRLayer(
            n_vars=state[0],
            use_dual=state[1],
            constraints=state[2]
        )

    def forward(self, associations, pt_in_a, x_in_b):
        r"""
        Compute the optimal pose given objects in frame a (`pt_in_a`) and objects in frame b
        (`pt_in_b`) and `associations` between the objects

        It is expected that `pt_in_a` and `pt_in_b` are tensors that are batch x num_objects x 2

        This solves the optimization problem:
            min_{T \in SE(2)} \sum_i \sum_j a_{ij} || p_{b,j} - T p_{a,i} ||^2
        where:
         - p_{a,i} is the ith point in `pt_in_a`
         - p_{b,j} is the jth point in `pt_in_b`
         - a_{ij} is `associations[i, j]`
        """

        cpu_associations = associations.cpu()
        Q = self._build_q_matrix(cpu_associations, pt_in_a, x_in_b, self._loss_coeffs)
        print(Q)
        solver_args = {
                # "solve_method": "Clarabel",
                "verbose": True
        }
        sol, _ = self._optimizer(Q, solver_args=solver_args)
        return sol[:, 1:, 0].to(associations.device)
