from typing import Union
from pathlib import Path
import logging

import tensorflow as tf


def export_onnx(model: tf.keras.Model,
                save_path: Union[str, Path],
                remove_dummy_opset: bool = True,
                dummy_prefix: str = 'args_0/'):
    """ 模型保存为onnx格式（保存路径）{save_path}.onnx

    :param model: tf.keras.Model, 待保存的keras模型
    :param save_path: str, 保存模型的路径
    :param remove_dummy_opset: bool, 是否移除多余的opset
    :param dummy_prefix: str, onnx的一个bug，会给从文件加载模型的节点加上args_0/前缀，需要移除
    """

    # 主流镜像缺少依赖，导入语句不要放在文件头
    import tf2onnx
    import onnx

    save_path = save_path + ".onnx"
    export_model = getattr(model, "onnx_export_model", None) or model
    onnx_model, _ = tf2onnx.convert.from_keras(export_model, output_path=save_path)

    if dummy_prefix:
        onnx_model = remove_dummpy_prefix(onnx_model, dummy_prefix)

    if remove_dummy_opset:
        onnx_model = remove_onnx_dummy_opset(onnx_model)
    onnx.save(onnx_model, save_path)
    logging.info("ONNX model is saved to %s.", save_path)


def remove_onnx_dummy_opset(onnx_model):
    """
    如果onnx proto有多个opset,只保留一个，否则NPU推理转换会失败
    :param onnx_model: ModelProto，输入模型
    :return onnx_model
    """
    # 主流镜像缺少依赖，导入语句不要放在文件头
    if len(onnx_model.opset_import) > 1:
        logging.info('Found more than one opset in ONNX model: %s. Only keep the first.', onnx_model.opset_import)
        del onnx_model.opset_import[1:]
        logging.info('Updated opset is %s', onnx_model.opset_import)
    return onnx_model


def remove_dummpy_prefix(onnx_model, prefix='args_0/'):
    """
    当前发现onnx对于从文件加载的tf model转换后输入节点会带上不必要的前缀（例如args_0
    如果onnx model的节点名字有,只保留第一个。否则NPU推理转换会失败
    :param onnx_model: ModelProto
    :param prefix: str, 需要移除的前缀
    :return onnx_model
    """
    # 主流镜像缺少依赖，导入语句不要放在文件头
    found = False

    for i, node_i in enumerate(onnx_model.graph.node):
        for j, input_j in enumerate(node_i.input):
            cur_name = input_j
            if cur_name.startswith(prefix):
                found = True
                logging.debug('Renaming %s to %s', onnx_model.graph.node[i].input[j], cur_name.split(prefix)[-1])
                onnx_model.graph.node[i].input[j] = cur_name.split(prefix)[-1]

    for i, input_i in enumerate(onnx_model.graph.input):
        cur_name = input_i.name
        if cur_name.startswith(prefix):
            found = True
            logging.debug('Renaming %s to %s', onnx_model.graph.input[i].name, cur_name.split(prefix)[-1])
            onnx_model.graph.input[i].name = cur_name.split(prefix)[-1]

    if found:
        logging.info('Found dummy prefix %s in onnx model. Rename finished.', prefix)
    return onnx_model
