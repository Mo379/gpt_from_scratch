import sys
from typing import Optional

import hydra
from omegaconf import DictConfig

class Singleton(type):
    """
    Singleton metaclass as described here:
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call function to create initial instance of class if it
        doesn't exist in _instances."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    """Config singleton class."""

    _dict_config_object: Optional[DictConfig] = None

    def __new__(cls) -> DictConfig:
        """Called every time a new instance is created
        (only once because it is a singleton)."""
        cls.initialize()
        return cls._dict_config_object

    @classmethod
    def initialize(cls, config_name="config", config_path="./") -> None:
        """Initializes or re-initializes the config."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path=config_path)

        # We still need this overwriting because the config value is
        # used in get_gpus() which is called in global scope.
        overrides = (
            ["raise_exception_if_gpus_not_available=false"]
            if "pytest" in sys.modules
            else []
        )

        new_config = hydra.compose(config_name=config_name, overrides=overrides)

        if cls._dict_config_object is None:
            cls._dict_config_object = new_config
        else:
            try:
                cls().update(new_config)
            except RecursionError as exc:
                raise RecursionError(
                    "Error during config initialization. "
                    "Very likely an issue with your JAX setup (e.g., GPU vs. CPU)."
                )

