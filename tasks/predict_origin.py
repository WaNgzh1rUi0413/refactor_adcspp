import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union

import keras.models
import pandas as pd
import numpy as np
import tensorflow as tf

from metrics.metric_builder import MetricBuilder
from utils import global_variables as gv
from utils.easydict import EasyDict
from utils.get_model_version import get_latest_train_model_path, get_hour_period
from utils.visualization.feature_embedding_visualizer import FeatureEmbeddingVisualizer
from utils.bits_analysis import analyze_bid_data_numeric, get_unique_filename
from .abstract_task import AbstractTask
from .evaluate import Evaluate


class Predict(AbstractTask):
    step = "predict"

    def __init__(
        self,
        cfg: EasyDict = None,
        model: keras.models.Model = None,
        test_dataset: tf.data.Dataset = None,
    ):
        """
        初始化预测任务。
        :param cfg: EasyDict, predict config, 测试配置项，比如
        {
            'use_keras_eval': False,
            'verbose': 'auto',
            'evaluator': {
                'metric': [
                    {'type': 'auc'}
                ]
            }
        }
        :param model: keras.models.Model, 需要执行测试的模型
        :param test_dataset: tf.data.Dataset, 测试数据集
        """
        self.cfg = cfg
        self.model = model
        self.test_dataset = test_dataset
        self.results = None
        self.visualization_cfg = self.cfg.get("visualization", {})
        self.is_model_loaded = False

        self._init_evaluators()

    def _init_evaluators(self):
        """初始化评估器(evaluators)的方法。"""
        self.evaluators = []
        self.evaluator_cfg = self.cfg.get("evaluator", {"metric": [{"type": "auc"}]})
        self.metrics = None
        self.use_keras_eval = False

        # 检查 evaluator 配置是否为列表
        if isinstance(self.evaluator_cfg, list):
            for cfg in self.evaluator_cfg:
                # 不支持 evaluator 列表 + use_keras_eval 组合
                if cfg.get("use_keras_eval", False):
                    raise NotImplementedError("use_keras_eval not supported for 'evaluator' list")
                if self.cfg.get("model_id", None):
                    cfg.model_id = self.cfg.model_id
                if self.cfg.get("date", None):
                    cfg.date = self.cfg.date
                self.evaluators.append(Evaluate(evaluator_config=cfg))
        else:
            self.use_keras_eval = self.evaluator_cfg.get("use_keras_eval", False)
            if self.use_keras_eval:
                self.metrics = self.evaluator_cfg.get("metric", [])
            self.evaluators.append(Evaluate(evaluator_config=self.evaluator_cfg))

    def run_keras_eval(self) -> Dict[str, float]:
        """
        调用 keras 的 evaluate 函数执行评估，返回包含各指标计算结果的字典。
        依赖模型已过编译，损失效率一般比自定义的更高，建议使用此函数执行评估。
        :return results: Dict[str, float]
        """
        verbose = self.cfg.get("verbose", "auto")
        self.model.compile(metrics=MetricBuilder.build(self.metrics))
        # evaluate 默认传入的是 tf.data.Dataset，需要是已经做过 batch 切分的
        results = self.model.evaluate(x=self.test_dataset, verbose=verbose)
        results = {name: res for name, res in zip(self.model.metrics_names, results)}
        return results

    def run_custom_eval(self, shuffle_feature=None):
        """
        自定义的评估实现，返回包含各指标计算结果的字典列表。先推理，再调用 evaluator 评估。
        :param shuffle_feature: 需要对预测集中打散的特征
        :return: results_list: list(Dict[str, float])
        """
        y_true, y_pred, group_columns = self.run_custom_inference(shuffle_feature)
        # 将预测结果导出到 CSV 文件
        output_file = self.cfg.get("output_file")
        if output_file:
            self.save_predictions_to_csv(y_true, y_pred, group_columns, output_file)
            # 对输出的 CSV 文件做数据分析
            analysis_file = self.cfg.get("analysis_file")
            if analysis_file:
                analyze_bid_data_numeric(output_file, get_unique_filename(analysis_file))
        results_list = []
        for evaluator in self.evaluators:
            results = evaluator.run(y_true, y_pred, group_columns)
            logging.info("Evaluation Results (%s): %s", evaluator.name, str(results))
            results_list.append(results)
        return results_list

    def get_group_column_names(self):
        """获取所有分组列名"""
        group_columns = []

        for evaluator in self.evaluators:
            for metric in evaluator.metrics:
                group_col = metric.get("config", {}).get("group_col", None)
                if group_col:
                    group_columns.append(group_col)
            # 用于分组下钻计算指标
            for groups in evaluator.metric_group_cols:
                group_columns += list(groups.keys())
        return list(set(group_columns))

    def run_custom_inference(self, shuffle_feature=None):
        """
        执行自定义的推理流程，返回 y_true_dict, y_pred_dict, group_columns。
        :param shuffle_feature: 需要对预测集中打散的特征
        :return y_true_dict: Dict[str, np.ndarray[N,]]
        :return y_pred_dict: Dict[str, np.ndarray[N,]]
        :return group_columns: Dict[str, np.ndarray[N,]]
        """
        y_true_dict = defaultdict(list)
        y_pred_dict = defaultdict(list)
        group_column_names = self.get_group_column_names()
        group_columns = {name: [] for name in group_column_names}

        for data in self.test_dataset:
            batch_features, batch_labels = data[:2]

            if shuffle_feature is not None and shuffle_feature in batch_features:
                batch_features[shuffle_feature] = tf.random.shuffle(batch_features[shuffle_feature])

            batch_pred = self.model(batch_features, training=False)

            # 将 batch_labels 和 batch_pred 存到 dict 中
            merge_outputs(batch_labels, y_true_dict, default_key=self.evaluator_cfg.get("y_true", "label"))
            merge_outputs(batch_pred, y_pred_dict, default_key=self.evaluator_cfg.get("y_pred", "pred"))

            batch_extra_input = data[2] if len(data) > 2 else {}

            for group_column_name in group_column_names:
                if group_column_name in batch_extra_input:
                    group_columns[group_column_name].append(batch_extra_input[group_column_name])
                elif group_column_name in batch_features:
                    group_columns[group_column_name].append(batch_features[group_column_name])
                elif group_column_name in batch_labels:
                    group_columns[group_column_name].append(batch_labels[group_column_name])
                else:
                    raise ValueError(f"Unknown group_column_name: {group_column_name}")

        # 转换为 NumPy 数组
        convert_and_compress_tensors(y_true_dict)
        convert_and_compress_tensors(y_pred_dict)
        convert_and_compress_tensors(group_columns, force_shape=True)

        return y_true_dict, y_pred_dict, group_columns

    def save_predictions_to_csv(self, y_true, y_pred, group_columns, output_file) -> None:
        """将预测结果保存到 CSV 文件"""
        output_columns = self.cfg.get("output_columns", [])
        try:
            # 处理数据：将二维 numpy 数组转换为数组列表
            all_data = convert_data_to_serializable_format([y_true, y_pred, group_columns])
            # 创建 DataFrame
            eval_df = pd.DataFrame(all_data)
            # 如果指定了要输出的列，则进行筛选
            if output_columns and output_columns != "all":
                available_columns = [col for col in output_columns if col in eval_df.columns]
                missing_columns = [col for col in output_columns if col not in eval_df.columns]
                if available_columns:
                    eval_df = eval_df[available_columns]
                if missing_columns:
                    logging.warning(
                        "The following columns do not exist and will be ignored: %s",
                        missing_columns,
                    )
            # 保存到 CSV 文件
            date = gv.MTP_TASK_CONFIG.date
            # 小时推理时，保存路径改成小时
            if gv.TOTAL_CONFIG.get("predict", {}).get("is_use_model_version", False):
                date = get_hour_period(gv.TOTAL_CONFIG, date)

            output_file = output_file.replace("#period_date_placeholder", date)
            check_path = "/".join(output_file.split("/")[:-1])
            if not os.path.exists(check_path):
                os.makedirs(check_path)
            eval_df.to_csv(output_file, index=False, header=False, encoding="utf-8")
            logging.info("The prediction results have been saved to file: %s", output_file)
            logging.info("The output columns are: %s", list(eval_df.columns))
        except (PermissionError, FileNotFoundError) as e:
            logging.error("Permission denied when saving to %s or directory not found for %s", output_file, output_file)

    def run_visualization(self):
        for visualizer_conf in self.visualization_cfg.get("visualizer"):
            if visualizer_conf.get("type", "FeatureEmbeddingVisualizer") == "FeatureEmbeddingVisualizer":
                visualizer = FeatureEmbeddingVisualizer(model=self.model, **visualizer_conf.get("params", {}))
                visualizer.run(self.test_dataset)

    def load_model(self, model_path: Union[str, Path] = None) -> None:
        """
        加载本地已有的权重文件
        :param model_path: Union[str, Path], 保存模型的路径。如果未指定，则使用默认路径
        :return: None
        """
        pth = model_path or gv.PATH_CONFIG.checkpoint_path
        if self.cfg.get("is_use_model_version", False):
            pth = get_latest_train_model_path(pth)
        logging.info("Try loading model from local checkpoint %s", pth)
        self.model.load_weights(pth)
        logging.info("Load local checkpoint success")
        self.is_model_loaded = True

    def run(self) -> None:
        """
        实现任务执行主流程
        :return: None
        """
        if not self.is_model_loaded:
            self.load_model()

        if self.visualization_cfg:
            self.run_visualization()
            if self.visualization_cfg.get("run_visualization_only", False):
                return

        if self.use_keras_eval:
            self.results = self.run_keras_eval()
        else:
            self.results = self.run_custom_eval()


def merge_outputs(batch_data, result_dict, default_key="label"):
    """
    根据 batch_data 的类型，将其组合并到 result_dict 中。
    如果 batch_data 是字典，将每个键值附加到 result_dict 中对应的列表中。
    否则，将 batch_data 附加到 result_dict 中指定的 default_key 对应的列表中。

    :param batch_data: (dict or any) 待合并的批次数据。如果是字典，将遍历其中的每个键值对；如果不是字典，将直接附加到指定的 key 下。
    :param result_dict: (dict) 用于存储合并结果的字典，每个键对应的值应为一个列表。
    :param default_key: (str, optional) 当 batch_data 不是字典时，指定合并到 result_dict 中的键。默认为 "label"。
    :return None: 该方法通过修改 result_dict 来实现功能，没有显式的返回值。
    """
    if isinstance(batch_data, dict):
        for key, value in batch_data.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
    else:
        if default_key not in result_dict:
            result_dict[default_key] = []
        result_dict[default_key].append(batch_data)


def convert_and_compress_tensors(dictionary, force_shape=False):
    """
    处理张量字典，将每个值转换为 NumPy 数组并进行维度压缩。
    :param dictionary: (dict) 包含张量列表的字典。Dict[str, list[Tensor[N,1]]]
    :param force_shape: (bool, optional) 是否强制检查张量的形状。默认为 False。
        - 当这里为 True 时，强制给予张量的形状是否符合预期，如果不符合，抛出 ValueError 错误。
        - 当这里为 False 时，不进行这种强制检查。
    """
    for key, tensor_list in dictionary.items():
        array = np.array(tf.concat(tensor_list, axis=0))
        dictionary[key] = array
        if array.ndim == 2:
            if array.shape[-1] == 1:
                dictionary[key] = array.squeeze(-1)
            elif force_shape:
                raise ValueError(
                    f"Column '{key}' has invalid shape {array.shape} "
                    f"(expected last dimension to be 1)"
                )
        elif array.ndim == 1:
            dictionary[key] = array
        elif force_shape:
            raise ValueError(
                f"Column '{key}' has {array.ndim} dimensions, "
                f"but only 1D or 2D arrays are supported"
            )


def convert_data_to_serializable_format(data_dicts) -> dict:
    """
    将输入的 numpy 数组转换为数组列表。
    :param data_dicts: 包含 y_true, y_pred 等数据的字典列表
    :return: 一个打开后的字典，所有值都已转换为一维列表形式。
    """
    all_data = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if hasattr(value, "shape") and len(value.shape) == 2:
                # 二维数组：转换为数组列表
                all_data[key] = value.tolist()
            else:
                # 一维数据：直接添加
                all_data[key] = value
    return all_data
