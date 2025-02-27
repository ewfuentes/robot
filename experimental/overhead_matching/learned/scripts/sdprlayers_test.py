
import unittest
import numpy as np

import common.torch.load_torch_deps
import torch
import scipy.optimize as opt

from sdprlayers.layers.sdprlayer import SDPRLayer


def eval_poly(x, params):
    out = None
    for i in range(len(params)):
        term = x ** i * params[i]
        if out is None:
            out = term
        else:
            out += term
    return out


def build_data_mat(p):
    Q_tch = torch.zeros((4, 4), dtype=torch.double)
    Q_tch[0, 0] = p[0]
    Q_tch[[1, 0], [0, 1]] = p[1] / 2
    Q_tch[[2, 1, 0], [0, 1, 2]] = p[2] / 3
    Q_tch[[3, 2, 1, 0], [0, 1, 2, 3]] = p[3] / 4
    Q_tch[[3, 2, 1], [1, 2, 3]] = p[4] / 3
    Q_tch[[3, 2], [2, 3]] = p[5] / 2
    Q_tch[3, 3] = p[6]

    return Q_tch


class SDPRLayersTest(unittest.TestCase):
    def test_sdpr_polynomial_example(self):
        """
        In this test case, we try to optimize a polynomial such that it has a global minimum
        at the desired location. The initialization has a local minimum near the desired location,
        but it also has a global minimum at some other location.
        """

        x0 = np.array([10.0, 2.6334, -4.3443, 0.0, 0.8055, -0.1334, 0.0389])

        desired_global_minumum = np.array([1.7, 7.3])

        print(f'initial polynomial: {x0} desired global min: {desired_global_minumum}')

        # Using the optimization framework provided by sdprlayers finds the global minimum of a
        # polynomial. It does this by framing the polynomial minimization problem as a quadratically
        # constrained quadratic program. That is we can pose the problem:
        #
        # min f(t)
        #   s.t. f is polynomial
        #
        # as:
        #
        # min_x x.T Q x
        #   s.t. x.T A_i x = 0
        #        x.T A_0 x = 1
        #
        # from some matrices Q and A_i. In particular, we can represent a degree n polynomial as the
        # summation of products of monomials that are up to degree n/2. The coefficients of the 
        # matrix Q represent the coefficients of the polynomial and the constraints A_i and A_0
        # ensure that the elements of `x` are powers of t, that is x.T = [1 t t^2 t^3].

        sdp_layer = SDPRLayer(
            n_vars=4,
            use_dual=False,
            constraints=[
                # This ensures that x1**2 = x2*x0
                np.array([
                    [  0,  0, 0.5, 0],
                    [  0, -1,   0, 0],
                    [0.5,  0,   0, 0],
                    [  0,  0,   0, 0]]),
                # This ensures that x3*x0 = x2*x1
                np.array([
                    [0,  0,  0, 1],
                    [0,  0, -1, 0],
                    [0, -1,  0, 0],
                    [1,  0,  0, 0]]),
                # This ensures that x1*x3 = x2**2
                np.array([
                    [0,   0,  0,   0],
                    [0,   0,  0, 0.5],
                    [0,   0, -1,   0],
                    [0, 0.5,  0,   0]]),
                # # This ensures that x0 = 1
                # This constraint is automatically added
                # np.array([
                #     [1, 0, 0, 0],
                #     [0, 0, 0, 0],
                #     [0, 0, 0, 0],
                #     [0, 0, 0, 0]]),
            ]
        )

        def sdp_outer_opt(params):
            p = params
            Q = build_data_mat(params)
            sol, _ = sdp_layer(Q, solver_args={'eps': 1e-10})
            x_min = 0.5 * (sol[1, 0] + sol[0, 1])
            y_min = eval_poly(x_min, p)
            loss = 0.5 * (desired_global_minumum[0] - x_min) ** 2 + 0.5 * (desired_global_minumum[1] - y_min)**2
            return x_min, y_min, loss

        params = torch.tensor(x0, requires_grad=True, dtype=torch.double)
        optimizer = torch.optim.Adam([params], lr=1e-2)

        loss = torch.tensor(1000)
        i = 0
        while loss.item() > 1e-8:
            optimizer.zero_grad()

            x_min, y_min, loss = sdp_outer_opt(params)
            x_min.retain_grad()

            loss.backward()
            optimizer.step()
            i += 1

        print(f'Optimized global min: ({x_min}, {y_min})')
        self.assertAlmostEqual(x_min.item(), desired_global_minumum[0], places=2)
        self.assertAlmostEqual(y_min.item(), desired_global_minumum[1], places=2)


    def test_local_polynomial_example(self):
        x0 = np.array([10.0, 2.6334, -4.3443, 0.0, 0.8055, -0.1334, 0.0389])

        desired_global_minumum = np.array([1.7, 7.3])

        print(x0, desired_global_minumum)
        # We expect that using a regular old optimization algorithm would get stuck in a local
        # minimum.
        def inner_opt(params, initial_guess):
            result = opt.minimize(
                lambda x: eval_poly(x, params),
                x0=initial_guess,
                method='BFGS',
                options={"eps": 1e-3})
            return result.x

        def outer_opt(params):
            inner_initial_guess = 2.0
            min_x = inner_opt(params, inner_initial_guess)
            min_y = eval_poly(min_x, params)

            return (desired_global_minumum[0] - min_x)**2 + (desired_global_minumum[1]-min_y)**2

        result = opt.minimize(outer_opt, x0=x0, method='CG')

        print(result)



if __name__ == "__main__":
    unittest.main()
    # main()
