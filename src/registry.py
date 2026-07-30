"""通用组件注册表 (Python 3.12+)"""

from collections.abc import Callable
from typing import Any


class Registry:
    """轻量级注册表，支持 Encoder / Head / Optimizer 的动态组装"""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: dict[str, type] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, module_name: str | None = None) -> Callable:
        """装饰器用法: @REGISTRY.register("my_module")"""

        def wrapper(cls: type) -> type:
            key = module_name or cls.__name__
            if key in self._module_dict:
                raise KeyError(f"[{self._name}] '{key}' already registered!")
            self._module_dict[key] = cls
            return cls

        return wrapper

    def build(self, cfg: dict[str, Any], **kwargs: Any) -> Any:
        """根据 {"type": "xxx", ...} 配置实例化对象"""
        cfg = cfg.copy()
        obj_type = cfg.pop("type", None)
        if obj_type is None:
            raise ValueError(f"[{self._name}] Config must contain 'type' key")
        if obj_type not in self._module_dict:
            raise KeyError(
                f"[{self._name}] '{obj_type}' not found. "
                f"Available: {list(self._module_dict.keys())}"
            )
        return self._module_dict[obj_type](**cfg, **kwargs)

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, modules={list(self._module_dict.keys())})"


ENCODER_REGISTRY = Registry("encoder")
HEAD_REGISTRY = Registry("head")
