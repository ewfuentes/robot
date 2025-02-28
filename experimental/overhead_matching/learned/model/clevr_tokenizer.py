import common.torch.load_torch_deps
import torch


def create_scene_tokens(scene_list_batch, vocabulary):
    """
    A batch is a list of scenes
    A scene is a list of items

    A vocabulary is a dictionary from strings to list of strings
    where the keys correspond to keys of an item and vocabulary[key]
    corresponds to all possible values that the key can take.

    This function assigns a unique token id for each object in the scene

    This function also returns a mask where elements that are true correspond
    to tokens that should be ignored
    """

    vocab_keys = sorted(vocabulary)
    num_scenes = len(scene_list_batch)
    max_tokens_per_scene = max([len(x) for x in scene_list_batch])
    tokens = torch.zeros((num_scenes, max_tokens_per_scene), dtype=torch.int32)
    mask = torch.ones((num_scenes, max_tokens_per_scene), dtype=torch.bool)

    for i, scene in enumerate(scene_list_batch):
        for j, item in enumerate(scene):
            embedding = 0
            for key in vocab_keys:
                options = vocabulary[key]
                embedding = embedding * len(options) + options.index(item[key])

            tokens[i, j] = embedding
            mask[i, j] = False

    return {"tokens": tokens, "mask": mask}


def create_position_embeddings(
    scene_list_batch, *, min_scale: float = 1e-3, scale_step: float = 2.0, embedding_size: int
):
    assert embedding_size % 4 == 0
    num_scenes = len(scene_list_batch)
    max_tokens_per_scene = max([len(x) for x in scene_list_batch])
    xy_pos = torch.zeros((num_scenes, max_tokens_per_scene, 2), dtype=torch.float32)
    for i, scene in enumerate(scene_list_batch):
        for j, item in enumerate(scene):
            xy_pos[i, j, 0] = item["3d_coords"][0]
            xy_pos[i, j, 1] = item["3d_coords"][1]

    out = torch.zeros(
        (num_scenes, max_tokens_per_scene, embedding_size), dtype=torch.float32
    )
    num_scales = embedding_size // 4
    for scale_idx in range(num_scales):
        embedding_idx_start = 4 * scale_idx
        scale = min_scale * scale_step**scale_idx / (2 * torch.pi)

        out[:, :, embedding_idx_start + 0] = torch.sin(xy_pos[:, :, 0] / scale)
        out[:, :, embedding_idx_start + 1] = torch.cos(xy_pos[:, :, 0] / scale)
        out[:, :, embedding_idx_start + 2] = torch.sin(xy_pos[:, :, 1] / scale)
        out[:, :, embedding_idx_start + 3] = torch.cos(xy_pos[:, :, 1] / scale)

    return out
