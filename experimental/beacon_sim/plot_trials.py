
import argparse
import os
import tqdm

from experimental.beacon_sim.sim_log_pb2 import SimLog

def load_results(results_path):
    results = {}
    for file_name in tqdm.tqdm(os.listdir(results_path)):
        with open(os.path.join(results_path, file_name), 'rb') as file_in:
            idx = int(file_name.split('.')[0])
            results[idx] = SimLog()
            results[idx].ParseFromString(file_in.read())
    return results

def main(results_path: str):
    results = load_results(results_path)
    print(len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', required=True)
    args = parser.parse_args()
    main(args.results_path)
