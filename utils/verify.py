import logging

import numpy as np
import pandas as pd
import tensorflow as tf


def validate_tensor_function(func):
    """
    Validate whether a function is Callable[[tf.Tensor, tf.Tensor], tf.Tensor].
    """
    test_true = tf.constant([1.0, 0.0])
    test_pred = tf.constant([1.0, 0.0])

    try:
        result = func(test_true, test_pred)
        if isinstance(result, tf.Tensor):
            return True
    except Exception:
        logging.debug("%s is not a valid tensor function", func.__name__)

    return False


def validate_array_function(func):
    """
    Validate whether a function is array-based.

    Callable[[np.ndarray, np.ndarray]] or Callable[[np.ndarray, np.ndarray, np.ndarray]].
    """
    test_true = np.array([1.0, 0.0])
    test_pred = np.array([1.0, 0.0])
    test_gid = np.array([1.0, 0.0])

    try:
        result = func(test_true, test_pred)
        return True
    except Exception:
        logging.debug("%s is not a valid array function", func.__name__)

    try:
        result = func(test_true, test_pred, test_gid)
        return True
    except Exception:
        logging.debug("%s is not a valid dataframe function", func.__name__)

    return False


def validate_dataframe_function(func):
    """
    Validate whether a function is a valid eval function.

    Call with eval_df and return metrics.
    """
    test_true = np.array([1.0, 0.0])
    test_pred = np.array([1.0, 0.0])
    test_group = np.array([1.0, 0.0])
    test_df = pd.DataFrame(
        {"y_pred": test_pred, "y_true": test_true, "test_group": test_group}
    )
    try:
        result = func(eval_df=test_df)
        return True
    except Exception:
        logging.debug("%s is not a valid dataframe function", func.__name__)

    try:
        result = func(eval_df=test_df, group_col="test_group")
        return True
    except Exception:
        logging.debug("%s is not a valid dataframe function", func.__name__)

    return False
