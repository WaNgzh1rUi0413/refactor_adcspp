import inspect
import copy
import logging
from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from core.registry import ClassRegistry
from networks.ranking_networks.dnn import DNN


class ModuleRegistry(ClassRegistry[Layer]):
    """Keras层注册器"""

    def __init__(self):
        super().__init__(base_class=Layer, registry_name="Layer")

        # 自动注册Keras原生层
        self._register_keras_native_layers()
        # 自动注册所有模块
        self.auto_register_from_module(module_path="networks", exclude_patterns=["network_builder"])

    def _register_keras_native_layers(self) -> None:
        """注册所有Keras原生层"""
        import tensorflow.keras.layers as keras_layers

        for name, obj in inspect.getmembers(keras_layers):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Layer)
                and obj != Layer
            ):
                self.register(name, obj)
                logging.debug("Registered Keras native layer: %s", name)


class CustomNetwork(Model):
    def __init__(self, network_config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = network_config
        self.module_configs = copy.deepcopy(self.config.get("seq_interactions", []))
        self.module_configs += self.config.get("field_interactions", [])
        self.module_configs += self.config.get("modules", [])
        # 校验配置更改为合法的DAG
        self._validate_dependency_graph()
        # 从当前网络(network)获取注册器
        self.registry = ModuleRegistry()
        # 初始化模块(layer/network)字典
        self.modules = {}
        # 构建模块
        self._build_modules()

    def call(self, inputs: Dict[str, tf.Tensor], training=None):
        module_outputs = {**inputs}
        for module_config in self.module_configs:
            module_name = module_config.get("name")
            module = self.modules.get(module_name)
            inputs_config = module_config.get("inputs", [])
            # 若模块输入为字典 [input_name: source]
            if isinstance(inputs_config, dict):
                module_input = self._resolve_module_input_dict(inputs_config, module_outputs)
            # 若模块输入为列表或元组 [source]
            elif isinstance(inputs_config, (list, tuple)):
                module_input = self._resolve_module_input_list(inputs_config, module_outputs)
            # 模块输入为单值 source
            elif isinstance(inputs_config, str):
                module_input = module_outputs.get(inputs_config)
            else:
                raise ValueError(f"Unsupported module input type: {type(inputs_config)}")

            try:
                module_output = module(module_input, training=training)
            except Exception as e:
                raise Exception(f"module:{module_name}, forward exception:{e}") from e

            self._put_module_output_into_module_outputs(module_name, module_output, module_outputs)

        # 根据source字段取出最终输出结果，输入为{'name': 'source'}格式的字典，输出为{'name': 'output_tensor'}格式的字典
        output_tensors = {}
        for output_name, output_source in self.config.get("outputs", {}).items():
            output_tensors[output_name] = module_outputs.get(output_source)
        return output_tensors

    def get_config(self):
        super_config = super().get_config()
        super_config.update({"network_config": self.config})
        return super_config

    def _build_modules(self):
        """构建模块"""
        for module_config in self.module_configs:
            module_name = module_config.get("name")
            module_class_name = module_config.get("type")
            module_params = module_config.get("params", {})

            module_class = self.registry.get(module_class_name)
            module = module_class(name=module_name, **module_params)
            self.modules[module_name] = module
        module_name_dict = {module_name: type(module).__name__ for module_name, module in self.modules.items()}
        logging.info("Built networks: %s", module_name_dict)

    @staticmethod
    def _put_module_output_into_module_outputs(module_name: str, module_output, module_outputs):
        # 将模块输出放进module_outputs中
        # 如果模块输出是dict，拆分到以模块名为module_name[key]
        if isinstance(module_output, dict):
            for key, out in module_output.items():
                module_outputs[f"{module_name}[{key}]"] = out
        # 如果module输出为list或tuple，则构多个输出命名为module_name[i]
        elif isinstance(module_output, (list, tuple)):
            for i, out in enumerate(module_output):
                module_outputs[f"{module_name}[{i}]"] = out
        # 否则，特指输出最终命名为module_name
        else:
            module_outputs[module_name] = module_output

    def _validate_dependency_graph(self):
        """
        校验配置是否为合法的有向无环图
        当检测到以下非法情况时抛出：
            - 非法模块名
            - 非法输入格式
            - 依赖的模块存在但未初始化
            - 循环依赖
        """

        # 初始化模块与模块输入列表
        all_modules = []
        for module_config in self.module_configs:
            # 检查模块名是否合法
            module_name = module_config.get("name")
            if not module_name:
                raise ValueError("Module name is required")
            if any(existing_module["name"] == module_name for existing_module in all_modules):
                raise ValueError(f"Duplicate module name: {module_name}")

            # 检查输入格式是否合法
            module_inputs = module_config.get("inputs")
            if not module_inputs:
                raise ValueError("Inputs of module_name is required")
            if not isinstance(module_inputs, (str, list, dict)):
                raise ValueError(f"Invalid inputs of {module_name}")

            all_modules.append(
                {
                    "name": module_name,
                    "inputs": module_inputs,
                }
            )

        # 遍历依赖关系，校验依赖顺序
        inited_modules = set()
        for module in all_modules:
            module_name = module["name"]
            module_inputs = module["inputs"]

            # 处理不同类型的输入
            if isinstance(module_inputs, str):
                dependencies = [module_inputs]
            elif isinstance(module_inputs, list):
                dependencies = module_inputs
            elif isinstance(module_inputs, dict):
                dependencies = list(module_inputs.values())

            dependencies = [d.split("[")[0] for d in dependencies]
            for dependency in dependencies:
                # 跳过外部输入（不在模块配置中的输入）
                if not any(m["name"] == dependency for m in all_modules):
                    continue

                if dependency not in inited_modules:
                    raise ValueError(f"Module {module_name} depends on {dependency} which is not inited")

                if dependency == module_name:
                    raise ValueError(f"Module {module_name} depends on itself")

            inited_modules.add(module_name)

    @staticmethod
    def _resolve_module_input_dict(
        inputs_config: Dict[str, str], module_outputs: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """解析输入某些dict"""
        tensors = {}
        for input_name, source in inputs_config.items():
            if source not in module_outputs:
                raise ValueError(f"Input '{source}' not found in current values")
            tensors[input_name] = module_outputs[source]
        return tensors

    @staticmethod
    def _resolve_module_input_list(
        inputs_config: List[str], module_outputs: Dict[str, tf.Tensor]
    ) -> List[tf.Tensor]:
        """解析输入某些list"""
        tensors = []
        for source in inputs_config:
            if source not in module_outputs:
                raise ValueError(f"Input '{source}' not found in current values")
            tensors.append(module_outputs[source])
        return tensors


class NetworkBuilder(object):
    @classmethod
    def build(cls, network_config: Dict[str, Any]) -> Layer:
        if network_config.get("type") == "DNN":
            return DNN(network_config)
        return CustomNetwork(network_config, name=network_config.get("name", "custom_network"))
