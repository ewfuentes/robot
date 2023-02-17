
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from typing import Optional

def get_points_of_interest(x_range_m, y_range_m, min_dist_away_m, num_points, rng: np.random.Generator):
    points = []
    while len(points) < num_points:
        x = rng.uniform(*x_range_m)
        y = rng.uniform(*y_range_m)

        should_skip = False
        for pt in points:
            dist = np.hypot(x-pt[0], y-pt[1])
            if dist < min_dist_away_m:
                should_skip = True
                break
        if not should_skip:
            points.append((x, y))
    return points

def launch_trials(points, output_dir, instance_set: Optional[list[int]]):
    for trial_idx, ((idx_1, pt_1), (idx_2, pt_2)) in enumerate(itertools.permutations(enumerate(points), 2)):
        if instance_set is not None and trial_idx not in instance_set:
            continue

        print(trial_idx, idx_1, pt_1, idx_2, pt_2)
        args = ['experimental/beacon_sim/run_trials',
                '--output', os.path.join(output_dir, f'trial_{trial_idx:06}.pb'),
                '--goal', f'{pt_1[0]},{pt_1[1]}',
                '--local_from_start', f'{pt_2[0]},{pt_2[1]},0.0',
                ]
        print(f'Running trial {trial_idx}...')
        subprocess.run(args)



def main(output_dir: str, instance_set: Optional[list[int]]):
    X_RANGE_M = (-5.0, 25.0)
    Y_RANGE_M = (-5.0, 25.0)
    NUM_POINTS = 35
    MIN_DIST_AWAY_M = 4.5
    rng = np.random.default_rng(seed=0)

    points = get_points_of_interest(X_RANGE_M, Y_RANGE_M, MIN_DIST_AWAY_M, NUM_POINTS, rng)

    os.makedirs(output_dir, exist_ok=True)

    launch_trials(points, output_dir, instance_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='Output directory', required=True)
    parser.add_argument('--instance_set', help='Run this set of instances', nargs="+", type=int)
    args = parser.parse_args()
    main(args.output_dir, args.instance_set)
