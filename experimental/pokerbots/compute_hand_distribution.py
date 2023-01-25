import argparse

import experimental.pokerbots.hand_evaluator_python as hep


def main(input_path, output_path):
    NUM_BINS = 8
    NUM_ROLLOUTS = 200
    MAX_ADDITIONAL_CARDS = 5
    print(input_path, output_path)
    # with open(input_path, "r") as file_in, open(output_path, "w") as file_out:
    #     for line in file_in:
    #         line = line.strip()
    #         hand, board = line[:4], line[4:]
    #         result = hep.estimate_hand_distribution(
    #             hand, board, NUM_BINS, NUM_ROLLOUTS, MAX_ADDITIONAL_CARDS
    #         )

    #         cards = [line[2*i, 2*i+2] for i in range(len(line)//2)]

    #         file_out.write(','.join(cards) + ',' + ','.join(result.distribution))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
