
import unittest

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import common.torch as torch
import scipy.optimize as opt

from sdprlayers.layers.sdprlayer import SDPRLayer

import numpy as np

class SDPRLayersTest(unittest.TestCase):
    def test_polynomial_example(self):
        """
        In this test case, we try to optimize a polynomial such that it has a global minimum
        at the desired location. The initialization has a local minimum near the desired location,
        but it also has a global minimum at some other location.
        """

        x0 = np.array([10.0, 2.6334, -4.3443, 0.0, 0.8055, -0.1334, 0.0389])

        desired_global_minumum = np.array([1.7, 7.3])

        print(x0, desired_global_minumum)

        # We expect that using a regular old optimization algorithm would get stuck in a local
        # minimum.
        def eval_poly(x, params):
            out = None
            for i in range(len(params)):
                term = x ** i * params[i]
                if out is None:
                    out = term
                else:
                    out += term
            return out


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

        x = np.linspace(-2.2, 2.4, 100)
        init_y = eval_poly(x, x0)
        y = eval_poly(x, result.x)
        plt.figure()
        plt.plot(x, init_y, label='init')
        plt.plot(x, y, label='scipy opt')
        plt.legend()

        # plt.show()

        # print(result)

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
            use_dual=False,
            n_vars=4,
            diff_qcqp=True,
            kkt_tol=1e-3,
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
                # np.array([
                #     [1, 0, 0, 0],
                #     [0, 0, 0, 0],
                #     [0, 0, 0, 0],
                #     [0, 0, 0, 0]]),
            ]
        )

        def sdp_outer_opt(params):
            p = params
            Q = torch.tensor([
                [      p[0], 1/2 * p[1], 1/3 * p[2], 1/4 * p[3]],
                [1/2 * p[1], 1/3 * p[2], 1/4 * p[3], 1/3 * p[4]],
                [1/3 * p[2], 1/4 * p[3], 1/3 * p[4], 1/2 * p[5]],
                [1/4 * p[3], 1/3 * p[4], 1/2 * p[5],       p[6]]], requires_grad=True)
            sol, x = sdp_layer(Q, solver_args={'eps': 1e-9})
            if x is None:
                return None
            x_min =0.5 * (sol[1, 0] +  sol[0, 1])
            y_min = eval_poly(x_min, p)
            loss = ((desired_global_minumum[0] - x_min) ** 2 + (desired_global_minumum[1] - y_min)**2)
            return loss

        initial_step = 1e-1
        step_factor = 0.5
        params = torch.tensor(x0, requires_grad=True)
        loss = sdp_outer_opt(params)
        loss.backward()
        print(params)
        iter_idx = 0
        while loss > 1e-4:
            # Run line search to ensure the next set of parameters is still feasible
            grad = params.grad
            run = True
            i = 0
            print(f"iter idx: {iter_idx:6d} loss: {loss.item()} params: {params.detach().numpy()} grad: {grad}")
            while run:
                new_params = params.detach().clone() - step_factor ** i * initial_step * grad
                new_params.requires_grad_(True)
                loss = sdp_outer_opt(new_params)
                print(f'\tline_search factor: {step_factor**i} loss: {loss.item()}')
                if loss is None:
                    continue
                params = new_params
                loss.backward()
                run = False
            i += 1
            iter_idx += 1



# def main():
#    import cvxpy
#    import numpy as np
#    x0 = np.array([10.0, 2.6334, -4.3443, 0.0, 0.8055, -0.1334, 0.0389])
#    # Q = cvxpy.Parameter((4, 4), symmetric=True)
#    # Q.value = np.array([
#    Q = np.array([
#        [      x0[0], 1/2 * x0[1], 1/3 * x0[2], 1/4 * x0[3]],
#        [1/2 * x0[1], 1/3 * x0[2], 1/4 * x0[3], 1/3 * x0[4]],
#        [1/3 * x0[2], 1/4 * x0[3], 1/3 * x0[4], 1/2 * x0[5]],
#        [1/4 * x0[3], 1/3 * x0[4], 1/2 * x0[5],       x0[6]]])
#
#    x = cvxpy.Variable((4, 4), symmetric=True)
#
#    constraints = [
#        # This ensures that x1**2 = x2*x0
#        cvxpy.trace(np.array([
#            [  0,  0, 0.5, 0],
#            [  0, -1,   0, 0],
#            [0.5,  0,   0, 0],
#            [  0,  0,   0, 0]]) @ x) == 0,
#        # This ensures that x3*x0 = x2*x1
#        cvxpy.trace(np.array([
#            [0,  0,  0, 1],
#            [0,  0, -1, 0],
#            [0, -1,  0, 0],
#            [1,  0,  0, 0]]) @ x) == 0,
#        # This ensures that x1*x3 = x2**2
#        cvxpy.trace(np.array([
#            [0,   0,  0,   0],
#            [0,   0,  0, 0.5],
#            [0,   0, -1,   0],
#            [0, 0.5,  0,   0]]) @ x) == 0,
#        # This ensures that x0 = 1
#        cvxpy.trace(np.array([
#            [1, 0, 0, 0],
#            [0, 0, 0, 0],
#            [0, 0, 0, 0],
#            [0, 0, 0, 0]]) @ x) == 1,
#        x >> 0,
#    ]
#    problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.trace(Q @ x)), constraints)
#    print(problem)
#    print(constraints)
#
#    result = problem.solve()
#
#    print(result, x.value)


if __name__ == "__main__":
    unittest.main()
    # main()
