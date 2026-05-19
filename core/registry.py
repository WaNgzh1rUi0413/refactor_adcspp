import importlib
import inspect
import logging
import pkgutil
from typing import Type, Generic, TypeVar, Dict, Optional, Callable, List

C = TypeVar('C', bound=type)
F = TypeVar('F', bound=Callable)
T = TypeVar('T', C, F)


class BaseRegistry(Generic[T]):
    """
    通用注册器基类
    """

    def __init__(self, registry_name: str):
        """
        初始化注册器
        :param registry_name: 注册器名称，用于日志记录
        :return:
        """
        self._registry: Dict[str, T] = {}
        self._registry_name = registry_name

    def register(self, name: str, obj: T) -> None:
        """
        注册对象
        :param name: 注册名称
        :param obj: 要注册的对象
        :return:
        """
        if name in self._registry:
            logging.warning(f'Overwriting existed registration: {name}')

        self._validate(obj)
        self._registry[name] = obj
        logging.debug('Registered %s: %s', self._registry_name, name)

    def get(self, name: str) -> T:
        """
        获取已注册的对象
        :param name: 注册名称
        :return: 注册的对象
        """
        if name not in self._registry:
            raise ValueError(f'{self._registry_name} {name} not registered')
        return self._registry[name]

    def list_all(self) -> Dict[str, T]:
        """获取所有注册项"""
        return self._registry.copy()

    def auto_register_from_module(self, module_path: str, exclude_patterns: Optional[List[str]] = None) -> None:
        """
        从指定模块中自动注册所有条件符合的对象
        :param module_path: 模块路径
        :param exclude_patterns: 要排除的模块名或解析时包含模式
        """
        exclude_patterns = exclude_patterns or []

        try:
            module = importlib.import_module(module_path)
            package_path = getattr(module, '__path__', [])
            package_name = module.__name__

            for _, submodule_name, _ in pkgutil.walk_packages(
                    path=package_path, prefix=f'{package_name}.'
            ):
                if any(pattern in submodule_name for pattern in exclude_patterns):
                    continue

                submodule = importlib.import_module(submodule_name)
                self._register_from_module(submodule, submodule_name)

        except Exception as e:
            logging.error(f"Error auto-registering from module {module_path}: {e}")

    def _register_from_module(self, module, module_name: str) -> None:
        """从指定模块中注册所有符合条件的对象"""
        for name, obj in inspect.getmembers(module):
            if self._is_valid(obj):
                self.register(name, obj)
                logging.debug('Auto-registered %s: %s from %s', self._registry_name, name, module_name)

    def _is_valid(self, obj: T) -> bool:
        """判断对象是否符合注册条件，由子类实现"""
        pass

    def _validate(self, obj: T) -> None:
        """验证对象是否符合注册条件，由子类扩展"""
        pass


class ClassRegistry(Generic[C], BaseRegistry[C]):
    """
    类注册器，用于注册指定基类的所有子类
    """

    def __init__(self, base_class: Type[C], registry_name: str):
        """
        初始化类注册器
        :param base_class: 要注册的类的基类
        :param registry_name: 注册器名称
        :return:
        """
        super().__init__(registry_name)
        self._base_class = base_class

    def _is_valid(self, obj: C) -> bool:
        """判断对象是否为符合条件的类"""
        return (
                inspect.isclass(obj)
                and issubclass(obj, self._base_class)
                and obj != self._base_class
        )

    def _validate(self, obj: C) -> None:
        """验证对象是否为符合条件的类"""
        if not self._is_valid(obj):
            raise ValueError(f"Object {obj} is not a valid subclass of {self._base_class}")


class FunctionRegistry(Generic[F], BaseRegistry[F]):
    """
    函数注册器，用于注册符合条件的函数
    """

    def __init__(self, registry_name: str, validator: Optional[Callable[[F], bool]] = None):
        """
        初始化函数注册器
        :param registry_name: 注册器名称
        :param validator: 函数验证器，用于检查函数是否符合注册条件
        :return:
        """
        super().__init__(registry_name)
        self._validator = validator or (lambda f: True)  # 默认接受所有函数

    def _is_valid(self, obj: F) -> bool:
        """判断对象是否为符合条件的函数"""
        return inspect.isfunction(obj) and self._validator(obj)

    def _validate(self, obj: F) -> None:
        """验证对象是否为符合条件的函数"""
        if not self._is_valid(obj):
            raise ValueError(f"Object {obj} is not a valid function")