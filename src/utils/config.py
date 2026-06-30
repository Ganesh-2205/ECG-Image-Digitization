"""
Configuration management module.
"""

from typing import Any, Dict
import os
import yaml


class Config:
    """
    Class-based configuration loader and accessor.
    Loads settings from YAML files.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        Initializes the configuration loader.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """
        Loads settings from the YAML file.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                self._config = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise RuntimeError(
                    f"Failed to parse YAML configuration: {exc}"
                ) from exc

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value by key.

        Args:
            key: The configuration key (supports dot-notation, e.g., 'paths.dataset_dir').
            default: Default value if key is not found.

        Returns:
            The value associated with the key.
        """
        keys = key.split(".")
        val: Any = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    @property
    def raw(self) -> Dict[str, Any]:
        """
        Returns the raw configuration dictionary.
        """
        return self._config
