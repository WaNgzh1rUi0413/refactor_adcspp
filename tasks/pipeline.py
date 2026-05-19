import logging
from abc import ABC, abstractmethod
from typing import List, Any

from core.distribute_strategy import StrategyBuilder
from utils.easydict import EasyDict
from utils.logger import log_decorator
from utils.timer import Timer, convert_time

from .abstract_task import AbstractTask
from .task_builder import TaskBuilder


class AbstractPipeline(ABC):
    """
    Pipeline base class.
    """

    def __init__(self, cfg: EasyDict = None, steps: List[str] = None):
        """
        Build a pipeline.

        :param cfg: EasyDict, global config
        :param steps: List[str], steps to execute
        """
        if not isinstance(cfg, EasyDict):
            raise ValueError("Please pass a valid 'pipeline_cfg'")
        if not steps:
            raise ValueError("Please pass a valid 'steps' argument")

        self.cfg = cfg
        self.steps = steps
        self._tasks = []
        self._pre_tasks = []
        self.is_built = False
        self.build_tasks()
        self._cache = {}

    @property
    def tasks(self):
        return self._tasks

    @abstractmethod
    def build_tasks(self):
        pass

    def add_task(self, task: AbstractTask):
        self._tasks.append(task)

    @abstractmethod
    def run(self):
        pass


class AdsRecPipeline(AbstractPipeline):
    def __init__(self, cfg: EasyDict = None, steps: List[str] = None):
        super().__init__(cfg, steps)

    @property
    def cache(self):
        return self._cache

    @property
    def model(self):
        return self._cache.get("model")

    @property
    def train_dataset(self):
        return self._cache.get("train_dataset")

    @property
    def valid_dataset(self):
        return self._cache.get("valid_dataset")

    @property
    def test_dataset(self):
        return self._cache.get("test_dataset")

    def set_task_model(self, task: AbstractTask) -> None:
        """
        Build model for task.

        :param task: AbstractTask instance
        :return: None
        """
        if getattr(task, "self_managed", False):
            return

        if task.step in ("train", "calibrate", "predict"):
            if not self.model:
                from models.model_builder import ModelBuilder

                model = ModelBuilder.build(self.cfg.model)
                if task.step in ["train", "calibrate"] or self.cfg.get(task.step).get(
                    "need_trainable_model"
                ):
                    model.build_trainable_model()

                self._cache["model"] = model

            task.model = self.model

    def set_task_model_with_strategy(
        self,
        task: AbstractTask,
        strategy: Any = None,
    ):
        """
        Build model with strategy.

        :param task: AbstractTask instance
        :param strategy: None|tf.distribute.Strategy
        """
        if strategy:
            with strategy.scope():
                self.set_task_model(task)
        else:
            self.set_task_model(task)

    def set_task_data(self, task: AbstractTask):
        """
        Build dataset for task.

        :param task: AbstractTask instance
        :return: None
        """
        if getattr(task, "self_managed", False):
            return

        if self.cfg.run.get("use_synth_data", False) or self.cfg.get(
            "use_synth_data", False
        ):
            logging.info("Using generator to synthesize random data")
            from inputs.generator.generator_builder import GeneratorBuilder

            generator = GeneratorBuilder.build(self.cfg.input)
            generator.generate()
            logging.info("Data generation success")

        use_small_data = self.cfg.run.get("use_small_data", False)
        small_data_batch = self.cfg.run.get("small_data_batch", 10)
        if use_small_data:
            logging.info("Using small data of %d batches", small_data_batch)

        if task.step == "train":
            if not self.train_dataset:
                logging.info("Creating train_dataset")
                from inputs.dataset_builder import DatasetBuilder

                train_dataset, valid_dataset = DatasetBuilder.build(self.cfg.input).build(
                    ["train", "valid"]
                )
                if use_small_data:
                    train_dataset = train_dataset.take(small_data_batch)
                    valid_dataset = valid_dataset.take(small_data_batch)

                self._cache.update(
                    {"train_dataset": train_dataset, "valid_dataset": valid_dataset}
                )
            task.train_dataset = self.train_dataset
            task.valid_dataset = self.valid_dataset

        if task.step == "calibrate":
            if not self.valid_dataset:
                logging.info("Creating valid_dataset")
                from inputs.dataset_builder import DatasetBuilder

                valid_dataset = DatasetBuilder.build(self.cfg.input).build(["valid"])[0]

                if use_small_data:
                    valid_dataset = valid_dataset.take(small_data_batch)

                self._cache["valid_dataset"] = valid_dataset

            task.valid_dataset = self.valid_dataset

        if task.step == "predict":
            if not self.test_dataset:
                logging.info("Creating test_dataset")
                from inputs.dataset_builder import DatasetBuilder

                test_dataset = DatasetBuilder.build(self.cfg.input).build(["test"])[0]

                if use_small_data:
                    test_dataset = test_dataset.take(small_data_batch)

                self._cache["test_dataset"] = test_dataset

            task.test_dataset = self.test_dataset

    def set_dependent_task(self, task: AbstractTask):
        """
        Set dependent task for current task.

        :param task: current task
        """
        if task.step.lower() == "mcp":
            for dependent_task in self.tasks:
                if dependent_task.step == "predict":
                    task.dependent_task = dependent_task
                    break
            else:
                raise ValueError(
                    "The mcp task is dependent on predict task. Please check!"
                )

    def build_tasks(self) -> None:
        """
        Build task pipeline.
        """
        if self.is_built:
            logging.info("Task chain already built. Skip")
            return

        self._tasks.clear()

        for step in self.steps:
            task_cfg = self.cfg.get(step)
            if task_cfg is None:
                logging.error("Task config of step '%s' is invalid", step)
                continue
            task = TaskBuilder.build(task_cfg)
            self.add_task(task)
            self.is_built = True

    def get_distribute_strategy(self) -> Any:
        """
        Build distribute strategy.
        """
        strategy = None
        strategy_name = self.cfg.get("train", {}).get(
            "distribute_strategy", "MirroredStrategy"
        )
        if self.cfg.run.get("num_device") > 1 and self.cfg.run.get("num_used_device") > 1:
            strategy = StrategyBuilder.build(strategy_name)
            logging.info("Train with %s strategy", strategy.__class__.__name__)
        return strategy

    @log_decorator
    def run(self):
        """
        Execute pipeline.
        """
        strategy = self.get_distribute_strategy()
        for task in self.tasks:
            task_name = task.__class__.__name__
            logging.info("task_name: %s", task.__class__.__name__)
            self.set_task_data(task)
            self.set_dependent_task(task)
            self.set_task_model_with_strategy(task, strategy)

            logging.info("Start running %s", task_name)
            with Timer() as t:
                task.run()
            logging.info("Finished %s after %s", task_name, convert_time(t.interval))
