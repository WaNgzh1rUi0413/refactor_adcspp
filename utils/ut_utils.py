import shutil
from typing import Dict

import numpy as np

from .config import constants
from . import global_variables as gv


def gen_random_features(
    num_sparse: int,
    num_samples=None,
    max_index=None,
    squeeze=False,
) -> Dict[str, np.ndarray]:
    """
    Generate random sparse features as test data.

    :param num_sparse: int, number of sparse features
    :param num_samples: int, batch size, defaults to gv.UT_RAND_DATA_BATCH_SIZE
    :param max_index: int, max index, defaults to gv.UT_FEATURE_MAX_INDEX
    :param squeeze: bool, return 1D array if True
    :return: Dict[str, np.ndarray]
    """
    if not num_samples:
        num_samples = gv.UT_RAND_DATA_BATCH_SIZE
    if not max_index:
        max_index = gv.UT_FEATURE_MAX_INDEX

    if squeeze:
        shape = (num_samples,)
    else:
        shape = (num_samples, 1)

    test_data = {
        f"sparse_{i}": np.random.randint(0, max_index, shape)
        for i in range(num_sparse)
    }
    return test_data


def clear_synth_data():
    """Remove synthetic data directory."""
    synth_data_dir = os.path.join(constants.LOCAL_OUTPUT_DIR, "data")
    if os.path.exists(synth_data_dir):
        shutil.rmtree(synth_data_dir)


def clear_checkpoint():
    """Remove checkpoint directory."""
    path = gv.PATH_CONFIG.checkpoint_path
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def gen_random_features_labels(
    num_sparse: int = 2,
    label_as_dict: bool = False,
    data_size=1000,
    label_pos_ratio=0.5,
):
    """
    Generate random feature/label pairs for tests.

    :param num_sparse: int, number of sparse features
    :param label_as_dict: bool, return label as dict if True
    :param data_size: int, dataset size
    :param label_pos_ratio: float, positive label ratio
    """
    inputs = gen_random_features(num_sparse, num_samples=data_size)
    labels = (np.random.rand(data_size) > label_pos_ratio).astype("int")
    if label_as_dict:
        labels = {"label": labels}
    return inputs, labels
