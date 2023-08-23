#!/usr/bin/env python3
"""
Tests for propagating the covariance matrix using the EKF with one-step updates
"""

import click
import numpy as np
import numpy.linalg as LA
from collections import defaultdict
import matplotlib.pyplot as plt

from experimental.beacon_sim.analysis.belief_propagation import BeliefPropagation
from experimental.beacon_sim.analysis.models import SingleIntegrator, DoubleIntegrator
from experimental.beacon_sim.analysis.star import scattering_from_transfer, transfer_from_scattering


def add_xy_to_axes(x, y, ax, name='', xlabel='', ylabel='', use_log=False):
    ax.semilogy(x, y, marker='o', label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()


@click.group()
def main():
    pass


@main.command(name="cond", help='Plot condition number for one-step update forms')
@click.argument("model", default="si")
@click.option("--num-steps", type=int, default=100, help='number of time steps')
@click.option("--dt", type=float, default=1, help='length of time step')
@click.option("--incr", type=int, default=1, help='increments to display in figures')
def run_cond(
    model,
    num_steps,
    dt,
    incr,
):
    #setup model
    if model=='si':
        P = np.identity(2)
        robot = SingleIntegrator(dt=dt)
    elif model=='di':
        P = np.identity(4)
        robot = DoubleIntegrator(dt=dt)
    else:
        print('error. invalid model')
        raise

    bp = BeliefPropagation(robot)   

    data={}

    #prop using gamma
    data['tf']=defaultdict(list)
    for i in range(num_steps):
        k=i*incr
        data['tf']['k'].append(k)

        gamma = bp.compute_tf(k=k)
        data['tf']['cond_gamma'].append(LA.cond(gamma))

        s = scattering_from_transfer(gamma)
        data['tf']['cond_s'].append(LA.cond(s))

        gamma_back = transfer_from_scattering(s)
        data['tf']['cond_gamma_back'].append(LA.cond(gamma_back))

    #prop using scatter
    data['star']=defaultdict(list)
    for i in range(num_steps):
        k=i*incr
        data['star']['k'].append(k)

        s = bp.compute_scatter(k=k)
        data['star']['cond_s'].append(LA.cond(s))

        gamma = transfer_from_scattering(s)
        data['star']['cond_gamma'].append(LA.cond(gamma))

        s_back = scattering_from_transfer(gamma)
        data['star']['cond_s_back'].append(LA.cond(s_back))

    #figures
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    add_xy_to_axes(data['star']['k'], data['star']['cond_gamma'],axes[0], name='$\kappa(\gamma(S))$', xlabel='number of steps', ylabel='condition number')
    add_xy_to_axes(data['star']['k'], data['star']['cond_s'],axes[0], name='$\kappa(S)$', xlabel='number of steps', ylabel='condition number')
    add_xy_to_axes(data['star']['k'], data['star']['cond_s_back'],axes[0], name='$\kappa(S(\gamma(S)))$', xlabel='number of steps', ylabel='condition number')
    axes[0].set_title('Condition Number $\kappa$ for Scattering Form',pad=20)

    add_xy_to_axes(data['tf']['k'], data['tf']['cond_gamma'],axes[1], name='$\kappa(\gamma)$', xlabel='number of steps', ylabel='condition number')
    add_xy_to_axes(data['tf']['k'], data['tf']['cond_s'],axes[1], name='$\kappa(S(\gamma))$', xlabel='number of steps', ylabel='condition number')
    add_xy_to_axes(data['tf']['k'], data['tf']['cond_gamma_back'],axes[1], name='$\kappa(\gamma(S(\gamma)))$', xlabel='number of steps', ylabel='condition number')
    axes[1].set_title('Condition Number $\kappa$ for Transfer Function Form',pad=20)

    plt.show()


@main.command(name="sens", help='Evaluate sensitivity to and from scatter form.')
@click.argument("model", default="si")
@click.option("--num-props", type=int, default=500)
@click.option("--dt", type=float, default=0.1)
def sens(
    model,
    num_props, #968, largest num props before fail for di
    dt,
):
    #setup model
    if model=='si':
        P = np.identity(2)
        robot = SingleIntegrator(dt=dt)
    elif model=='di':
        P = np.identity(4)
        robot = DoubleIntegrator(dt=dt)
    else:
        print('error. invalid model')
        raise

    bp = BeliefPropagation(robot)

    P_ekf = bp.prop(P, k=num_props)
    P_scatter = bp.prop_scatter(P, k=num_props)
    P_tf = bp.prop_tf(P, k=num_props)

    gamma = bp.prop_tf(P, k=num_props, return_tf=True)
    print('cond(gamma):',LA.cond(gamma)) 

    rand_numbers = np.random.rand(1000)
    rand_numbers = rand_numbers/np.sum(rand_numbers)
    gamma_expected = np.zeros_like(gamma)
    for r in rand_numbers:
        gamma_expected += r*gamma
    P_expected = bp.apply_tf_to_cov(P,gamma)

    diff_ekf_scatter = LA.norm(P_ekf - P_scatter)
    diff_ekf_tf = LA.norm(P_ekf - P_tf)
    diff_expected = LA.norm(P_scatter - P_expected)
    print('||P_ekf - P_scatter||      =', diff_ekf_scatter)
    print('||P_ekf - P_tf||           =', diff_ekf_tf)
    print('||P_scatter - P_expected|| =', diff_expected)


if __name__ == "__main__":
    main()
