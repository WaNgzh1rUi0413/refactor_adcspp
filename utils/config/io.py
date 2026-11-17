import os
import stat
import json
import logging

import yaml

from utils import global_variables as gv


def read_yml(file_path, use_e=False):
    """
    :param file_path: 文件路径
    :param use_e: yml文件中是否含有科学计数法
    :return:
    """
    with open(file_path, "r", encoding="UTF-8") as yml_file:
        if not use_e:
            result = yaml.safe_load(yml_file)
        else:
            # DataOps环境没有ruamel包
            from ruamel.yaml import YAML

            yaml_obj = YAML(typ="safe")
            result = yaml_obj.load(yml_file)
    return result


def read_json(json_path: str):
    """
    读取json格式文件
    :param json_path: 文件路径
    :return:
    """
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.loads(f.read())
    return result


def save_json(json_path: str, data: dict):
    """
    将data保存到json文件
    :param json_path: 文件路径
    :param data: 保存数据
    :return:
    """
    flags = os.O_RDWR | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(json_path, flags, modes), mode="w") as f:
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


def get_specified_feature_map_dict(feature_name: str) -> dict:
    """
    获取指定特征在 featureMap 文件中的 valid 值域映射字典
    :param feature_name: 多场景用于区分子网特征
    :return: 指定特征在 featureMap 文件中的 id 映射字典
    """
    if "_" in feature_name or "*" in feature_name:
        raise NotImplementedError("Cross-feature not supported yet.")
    feature_map_path = gv.PATH_CONFIG.feature_map_path
    logging.info("feature_map_path: %s", feature_map_path)
    feature_map_dict = read_json(feature_map_path)

    spec_feature_val2id = {}
    # featureMap.json为 feature_name, val:id格式
    for feat_val, _id in feature_map_dict.items():
        arr = feat_val.split(":")
        if arr[0] == feature_name and len(arr) == 2:
            (_, val) = arr
            spec_feature_val2id[val] = int(_id)

    if not spec_feature_val2id:
        raise ValueError(f"No feature {feature_name} found in feature map file.")
    logging.info("feature: %s val2id: %s", feature_name, spec_feature_val2id)
    return spec_feature_val2id
