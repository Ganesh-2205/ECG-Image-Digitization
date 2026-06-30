"""
Logging configuration utility.
"""

import logging
import os
import sys
from typing import Optional


class SetupLogger:
    """
    Initializes and configures the standard logging system.
    """

    @staticmethod
    def get_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Configures and returns a logger instance.

        Args:
            name: Name of the logger.
            log_file: Optional file path to write log outputs.
            level: Logging level (e.g. logging.INFO).

        Returns:
            A configured Logger instance.
        """
        logger = logging.getLogger(name)
        
        # Avoid duplicate handlers if already configured
        if logger.hasHandlers():
            return logger
            
        logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (Optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger
