"""
Utility functions for file system interactions.
"""

import json
import os
from typing import Any, Dict, List, Union
import yaml


class FileUtils:
    """
    Utility helper class for managing files, directories, and data persistence.
    """

    @staticmethod
    def ensure_dir(directory_path: str) -> None:
        """
        Creates a directory if it does not exist.

        Args:
            directory_path: Path of the directory.
        """
        os.makedirs(directory_path, exist_ok=True)

    @staticmethod
    def get_files_with_extension(
        directory: str, extensions: Union[str, List[str]]
    ) -> List[str]:
        """
        Finds all files in a directory matching specific extensions.

        Args:
            directory: Path to search in.
            extensions: List of extensions or a single extension string (e.g. '.png').

        Returns:
            Sorted list of absolute or relative file paths.
        """
        if isinstance(extensions, str):
            extensions = [extensions]
            
        extensions = [ext.lower() for ext in extensions]
        
        if not os.path.exists(directory):
            return []

        matched_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    matched_files.append(os.path.join(root, file))
                    
        return sorted(matched_files)

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        """
        Saves data as a structured JSON file.

        Args:
            data: Dictionary of metrics or configurations.
            file_path: Save destination path.
        """
        FileUtils.ensure_dir(os.path.dirname(file_path))
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def save_yaml(data: Dict[str, Any], file_path: str) -> None:
        """
        Saves data as a YAML file.

        Args:
            data: Dictionary data.
            file_path: Save destination path.
        """
        FileUtils.ensure_dir(os.path.dirname(file_path))
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
