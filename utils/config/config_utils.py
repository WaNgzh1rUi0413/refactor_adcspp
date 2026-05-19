import copy
import logging
from typing import Dict, Union

from utils.easydict import EasyDict
from .defaults import DEFAULTS
from .io import read_yml


def get_cfg_from_file(path: str, merge_default=True):
    """
    Load config file and optionally merge with defaults.

    Args:
        path: Config file path.
        merge_default: Whether to merge missing fields with defaults.

    Returns:
        EasyDict of merged config.
    """
    user_cfg = EasyDict(read_yml(path))
    if not merge_default:
        return user_cfg
    default_cfg = get_default_cfg()
    _merge_a_into_b(user_cfg, default_cfg)
    return default_cfg


def _merge_a_into_b(src: Union[EasyDict, Dict], dst: Union[EasyDict, Dict]):
    """
    Merge src into dst recursively.

    Args:
        src: Source config dictionary.
        dst: Destination config dictionary.
    """
    if not isinstance(src, EasyDict):
        return

    for k, v in src.items():
        # src must specify keys that are in dst
        if k not in dst:
            raise KeyError("{} is not a valid config key".format(k))

        # the types must match, too
        old_type = type(dst[k])
        if old_type is not type(v):
            raise ValueError(
                "Type mismatch ({} vs {}) for config key: {}".format(
                    type(dst[k]),
                    type(v),
                    k,
                )
            )

        # recursively merge dicts
        if isinstance(v, EasyDict):
            try:
                _merge_a_into_b(src[k], dst[k])
            except Exception:
                logging.error("Error under config key: {}".format(k))
                raise
        else:
            dst[k] = v


def get_default_cfg():
    """
    Return a copy of default configs. Do not modify defaults directly.

    Returns:
        EasyDict of default configs.
    """
    return copy.deepcopy(DEFAULTS)
