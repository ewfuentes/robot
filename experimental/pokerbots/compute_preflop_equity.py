
import argparse
import itertools
import pickle

import experimental.pokerbots.hand_evaluator_python as hep


def main(output_path):
    ranks = '23456789TJQKA'
    suits = 'shcd'
    cards = [''.join(card) for card in itertools.product(ranks, suits)]
    equity = {}
    for i, hole_cards in enumerate(itertools.combinations(cards, 2)):
        hole_cards = ''.join(hole_cards)
        result = hep.evaluate_expected_strength(hole_cards, 'random', '', 0.1)
        equity[hole_cards] = result.strength

    with open(output_path, 'wb') as file_out:
        pickle.dump(equity, file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()

    main(args.output)
