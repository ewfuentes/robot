
import argparse
import sys
import os
import numpy as np

import matplotlib
import matplotlib.style
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt

from experimental.beacon_sim.rollout_statistics_pb2 import AllStatistics, RolloutStatistics
from common.math.matrix_pb2 import Matrix as ProtoMat

def load_results(results_path):
    with open(results_path, 'rb') as file_in:
        results = AllStatistics()
        results.ParseFromString(file_in.read())
    return results

def matrix_from_proto(mat: ProtoMat):
    out  = np.zeros((mat.num_rows, mat.num_cols))
    for r in range(mat.num_rows):
        for c in range(mat.num_cols):
            out[r, c] = mat.data[r * mat.num_cols + c]
    return out

def compute_covariance_size(result: RolloutStatistics):
    full_cov = matrix_from_proto(result.final_step.posterior.cov)
    robot_cov = full_cov[:3, :3]
    return np.linalg.det(robot_cov)

def main(results_path: str):
    results = load_results(results_path)
    covariance_sizes = compute_covariance_size(results)
    plt.figure()
    plt.hist(covariance_sizes, bins=1000)
    plt.savefig('/tmp/plot.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', required=True)
    args = parser.parse_args()
    main(args.results_path)
