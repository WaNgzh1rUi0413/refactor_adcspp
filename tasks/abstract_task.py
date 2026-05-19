from abc import ABC, abstractmethod


class AbstractTask(ABC):
    """
    Task base class.

    Each task must define a step attribute. The step is a stage of the pipeline.
    The config for a step should live under the step name in the config file,
    e.g. train/test/mcp.
    """

    step = None

    @abstractmethod
    def run(self) -> None:
        """Define the task execution process."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Allow task instances to be called like a function to run the task."""
        self.run(*args, **kwargs)
