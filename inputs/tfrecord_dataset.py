import json
import logging
import os
from collections import OrderedDict
from typing import List, Dict
import tensorflow as tf
from utils.config.constants import DPP_MODEL_TRAIN_CONFIG, SEQ_MAX_LENGTH_KEY, MAX_INDEX_KEY, DPP_DATA_SUBDIR, \
    CONFIG_CROSS_SEP, ACTUAL_CROSS_SEP, LOCAL_OUTPUT_DIR, MTP_DATA_DIR
from utils.easydict import EasyDict
from utils.get_model_version import get_latest_dataset_period
from utils.global_variables import DEFAULT_INPUT_TYPE_MAPPING, MTP_TASK_CONFIG, IS_MTP
from .dataset import Dataset
from utils import global_variables as gv


class TFRecordDataset(Dataset):
    input_props = {}

    def __init__(self, input_conf: EasyDict = None):
        super(TFRecordDataset, self).__init__()
        self.input_column_conf = input_conf.input_column
        self.date = MTP_TASK_CONFIG.date
        self.load_conf = input_conf.load_conf
        self.compression_type = self.load_conf.get("compression_type", None)
        self.cycle_length = self.load_conf.cycle_length
        self.block_length = self.load_conf.block_length
        self.num_parallel_calls = self.load_conf.num_parallel_calls
        self.shuffle_buffer_size = self.load_conf.shuffle_buffer_size
        self.shuffle_seed = self.load_conf.shuffle_seed
        self.batch_size = self.load_conf.batch_size
        self.prefetch_size = self.load_conf.prefetch_size
        self.buffer_size = self.load_conf.buffer_size
        self.parallel_load_method = self.load_conf.parallel_load_method
        self.cfg = input_conf

        if IS_MTP:
            self._init_mtp_conf()
        else:
            self._init_local_conf()

        self.feature_desc = self._get_feature_desc()
        # input_names_per_type形如{sparse_feature: [aid, age],...}
        self.input_names_per_type: Dict[str, List]
        self.input_props, self.input_names_per_type = self._parse_input_conf()
        TFRecordDataset.input_props = self.input_props

    def _init_mtp_conf(self):
        """
        初始化DPP相关参数
        :return:
        """
        period = self.date
        if gv.TOTAL_CONFIG.get("predict", {}).get("is_use_model_version", False):
            period = get_latest_dataset_period(gv.TOTAL_CONFIG, period)

        model_train_config_path = os.path.join(MTP_DATA_DIR, MTP_TASK_CONFIG.scene_name, MTP_TASK_CONFIG.model_id,
                                               "config", period, DPP_MODEL_TRAIN_CONFIG)
        # 存储maxIndex和seqMaxLength
        model_train_config_dict = json.loads(open(model_train_config_path, 'r').read())
        logging.info(msg='modelTrain config content is %s', args=json.dumps(model_train_config_dict, indent=2))
        self.seq_max_length_dict = model_train_config_dict.get(SEQ_MAX_LENGTH_KEY)
        self.max_index_dict = model_train_config_dict.get(MAX_INDEX_KEY)
        self.data_path = {
            k: os.path.join(MTP_DATA_DIR, MTP_TASK_CONFIG.scene_name, MTP_TASK_CONFIG.model_id, "data", period, k)
            for k in DPP_DATA_SUBDIR
        }

    def _init_local_conf(self):
        """
        初始化本地训练相关参数
        :return:
        """
        model_train_config_path = os.path.join(LOCAL_OUTPUT_DIR, "config", DPP_MODEL_TRAIN_CONFIG)
        model_train_config_dict = json.loads(open(model_train_config_path, 'r').read())
        logging.info(msg='modelTrain config content is %s', args=json.dumps(model_train_config_dict, indent=2))
        self.seq_max_length_dict = model_train_config_dict.get(SEQ_MAX_LENGTH_KEY)
        self.max_index_dict = model_train_config_dict.get(MAX_INDEX_KEY)
        self.data_path = {k: os.path.join(LOCAL_OUTPUT_DIR, "data", k) for k in DPP_DATA_SUBDIR}

    def _parse_input_conf(self):
        """
        获取输出特征相关的属性字典
        :return: Dict
        """
        input_props = {}
        input_names_per_type = {}
        for input_column_type, input_column_conf_list in self.input_column_conf.items():
            input_names_per_type[input_column_type] = []

            for input_column_conf in input_column_conf_list:
                if isinstance(input_column_conf, str):
                    input_name = input_column_conf
                else:
                    input_name = input_column_conf["name"]
                input_names_per_type[input_column_type].append(input_name)
                input_props[input_name] = {}
                input_props[input_name]["name"] = input_name
                input_props[input_name]["input_type"] = input_column_type
                input_props[input_name]["seq_max_length"] = self.seq_max_length_dict.get(
                    input_name.replace(CONFIG_CROSS_SEP, ACTUAL_CROSS_SEP))
                joint_name = input_column_conf.get("joint_name") if not isinstance(input_column_conf, str) else None
                input_props[input_name]["joint_name"] = joint_name
                input_props[input_name]["max_index"] = self.max_index_dict.get(
                    joint_name or input_name.replace(CONFIG_CROSS_SEP, ACTUAL_CROSS_SEP))

        return input_props, input_names_per_type

    def _get_feature_desc(self):
        """
        获取所有输入字段(特征、标签等)和对应的tf.io.FixedLenFeature对应的字典，FixedLenFeature指定了如何读取该字段，
        用于反序列化tfrecord文件，字典可以包括输入文件中的全部字段，也不需要关注顺序
        :return: 输入字段名称和tf.io.FixedLenFeature对象的字典
        """
        # 注意：对于交叉特征需要将特征名称中的"#"替换为"__"
        feature_desc = {}
        for input_column_type, input_column_conf_list in self.input_column_conf.items():
            # 获取输入类型(sparse/dense..)默认的数据类型(int64, float32)
            default_data_type = DEFAULT_INPUT_TYPE_MAPPING.get(input_column_type)

            for input_column_conf in input_column_conf_list:
                if isinstance(input_column_conf, str):
                    input_column_conf = {"name": input_column_conf}
                input_name = input_column_conf.get("name")
                data_type = input_column_conf.get("data_type", default_data_type)
                fill_value = input_column_conf.get("fill_value")

                if not data_type:
                    raise ValueError("Invalid input type...")

                # label的序列长度并没有被保存在modelTrain.json文件中，需要特殊处理
                if input_column_type != "labels" and input_name != "column_id":
                    # 需要考虑对交叉特征的特殊处理，将"#"替换为"__"
                    seq_max_length = self.seq_max_length_dict.get(input_name.replace(CONFIG_CROSS_SEP,
                                                                                      ACTUAL_CROSS_SEP))
                else:
                    seq_max_length = input_column_conf.get("seq_max_length", 1)
                feature_desc[input_name.replace(CONFIG_CROSS_SEP, ACTUAL_CROSS_SEP)] = tf.io.FixedLenFeature(
                    dtype=data_type, shape=seq_max_length,
                    default_value=fill_value)
        return feature_desc

    def _get_dataset(self, file_list: List = None, data_type: str = None) -> tf.data.Dataset:
        """
        从给定的文件列表获取dataset，根据data_type不同对数据的处理方式会有所不同，例如是否需要shuffle等
        :param file_list:
        :param data_type:
        :return:
        """

        def paraller_interleave_func(x):
            return tf.data.TFRecordDataset(
                x,
                compression_type=self.compression_type,
                buffer_size=self.buffer_size,
                num_parallel_reads=self.num_parallel_calls or tf.data.AUTOTUNE
            )

        tf_data_file_list = tf.data.TFRecordDataset.from_tensor_slices(file_list)
        if self.parallel_load_method == "interleave":
            dataset = tf_data_file_list.prefetch(tf.data.AUTOTUNE).interleave(
                paraller_interleave_func,
                cycle_length=self.cycle_length or tf.data.AUTOTUNE,
                num_parallel_calls=self.num_parallel_calls or tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)
        elif self.parallel_load_method == "parallel_interleave":
            dataset = tf_data_file_list.apply(
                tf.data.experimental.parallel_interleave(
                    paraller_interleave_func,
                    cycle_length=self.cycle_length,
                    sloppy=True if data_type == "train" else False,
                    buffer_output_elements=self.buffer_size,
                    prefetch_input_elements=self.prefetch_size
                )
            )
        else:
            raise ValueError("Invalid parallel_load_method...")
        if data_type == "train":
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.shuffle_seed)
        dataset = dataset.batch(self.batch_size, drop_remainder=self.cfg.get('drop_remainder', False))
        dataset = dataset.map(self._parse_example, num_parallel_calls=self.num_parallel_calls or tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def build(self, dataset_type_list: List, dataset_path_list: List = None):
        """
        获取在所有task处理过程中需要的所有dataset，需要已知所需datasets的文件路径
        :return: 字典形式返回的所有task处理过程中需要的datasets
        """
        datasets = []
        for index, dataset_type in enumerate(dataset_type_list):
            if dataset_path_list:
                file_list = self._get_tfrecord_file_list_from_path(dataset_path_list[index])
            else:
                logging.info(msg='Loading TFRecords for %s from path %s',
                             args=(dataset_type, self.data_path[dataset_type]))
                file_list = self._get_tfrecord_file_list_from_path(self.data_path[dataset_type])
            if dataset_type == 'test' and gv.TOTAL_CONFIG.get("predict", {}).get("is_use_model_version", False):
                # 只有小时级推理场景
                latest_period = get_latest_dataset_period(gv.TOTAL_CONFIG, self.date)
                test_data_path = os.path.join(MTP_DATA_DIR, MTP_TASK_CONFIG.scene_name, MTP_TASK_CONFIG.model_id,
                                              "data", latest_period, dataset_type)
                file_list = self._get_tfrecord_file_list_from_path(test_data_path)
            logging.info(msg="file_list is %s, ", args=file_list)
            logging.info(msg="dataset_type is %s, ", args=dataset_type)
            dataset = self._get_dataset(file_list, dataset_type)
            datasets.append(dataset)
        return datasets

    def _parse_example(self, ser_data: tf.Tensor):
        """
        获取解析tfrecord数据文件的函数

        当前支持解析三种类型的数据：

        * 所有作为输入参与模型训练的特征
        * 所有模型训练需要的标签，可以包含多个，以字典的形式返回
        * 所有不作为模型输入，但是参与模型训练的特征，如样本权重等

        :param ser_data: 待解析的数据
        :return: 解析后的三类数据
        """
        examples = tf.io.parse_example(serialized=ser_data, features=self.feature_desc)

        labels = OrderedDict()
        label_names = self.input_names_per_type.get("labels", [])
        for label_name in label_names:
            labels[label_name] = examples.pop(label_name)

        extra_inputs = OrderedDict()
        extra_feature_names = self.input_names_per_type.get("extra_input", [])
        for extra_feature_name in extra_feature_names:
            extra_inputs[extra_feature_name] = examples.pop(extra_feature_name)

        return examples, labels, extra_inputs

    @staticmethod
    def _get_tfrecord_file_list_from_path(path: str = None):
        """
        从给定目录读取所有的tfrecord文件
        :param path: 文件目录
        :return: 目录下tfrecord文件的路径
        """
        files = []
        for file in os.listdir(path):
            if file.startswith('_') or file.startswith('.'):
                continue

            if file.endswith(".tfrecord.gz"):
                files.append(os.path.join(path, file))

        if not files:
            raise ValueError("input file is empty, please check if local file is empty or not")
        return files