
import argparse

import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np

import common.torch as torch

from experimental.overhead_matching.learned.model import (
    clevr_transformer,
    clevr_tokenizer
)
from experimental.overhead_matching.learned.data import (
    clevr_dataset
)
from experimental.overhead_matching.learned.scripts import (
    train_clevr_transformer as tct
)

EMBEDDING_SIZE=128

def load_model(model_path: Path):
    config = clevr_transformer.ClevrTransformerConfig(
        token_dim=EMBEDDING_SIZE,
        vocabulary_size=96,
        num_encoder_heads=4,
        num_encoder_layers=8,
        num_decoder_heads=4,
        num_decoder_layers=8,
        output_dim=4,
        predict_gaussian=True
    )

    model = clevr_transformer.ClevrTransformer(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

def main(model_path: Path, training_dataset: Path, dataset_path: Path):
    model = load_model(model_path).cuda().eval()

    vocabulary = clevr_dataset.ClevrDataset(training_dataset).vocabulary()
    eval = clevr_dataset.ClevrDataset(dataset_path)
    loader = clevr_dataset.get_dataloader(eval, batch_size=128)

    rng = np.random.default_rng(1024)
    ego_from_worlds = []
    results = []
    for batch in loader:
        batch = batch["objects"]
        ego_from_worlds.append(tct.sample_ego_from_world(rng, len(batch)))
        inputs = tct.clevr_input_from_batch(batch, vocabulary, EMBEDDING_SIZE, ego_from_worlds[-1])
        query_tokens = None
        query_mask = None

        results.append(model(inputs, query_tokens, query_mask))

    
    
    plt.figure()
    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--training_dataset", required=True)
    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    main(Path(args.model), Path(args.training_dataset), Path(args.dataset))
