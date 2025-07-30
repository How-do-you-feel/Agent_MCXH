from abc import ABCMeta, abstractmethod
from typing import Any, Tuple
import copy

from ..meta import ToolMeta, Parameter

class BaseTool(metaclass=ABCMeta):
    def __init__(self, toolmeta=None):
        if toolmeta is None:
            toolmeta = ToolMeta(name=self.__class__.__name__)
        self.toolmeta = toolmeta
        self._is_setup = False

    @property
    def name(self) -> str:
        return self.toolmeta.name

    @property
    def description(self) -> str:
        return self.toolmeta.description or ""

    @property
    def inputs(self) -> Tuple[Parameter, ...]:
        return self.toolmeta.inputs or ()

    @property
    def outputs(self) -> Tuple[Parameter, ...]:
        return self.toolmeta.outputs or ()

    def setup(self):
        """Initialize the tool before first use. Override this method to load models, etc."""
        pass

    def __call__(self, *args, **kwargs):
        if not self._is_setup:
            self.setup()
            self._is_setup = True
        return self.apply(*args, **kwargs)

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Implement the actual tool functionality here"""
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}(toolmeta={self.toolmeta})'