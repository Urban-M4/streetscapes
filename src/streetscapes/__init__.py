"""Streetscapes python package for large-scale analysis of street-level imagery."""

import os

from dotenv import load_dotenv

from streetscapes.config import CFG
from streetscapes.utils import logger


def get_env(key: str):
    """Get environment variable from user's env."""
    load_dotenv()
    value = os.getenv(key, None)
    if value is not None:
        return value
    raise KeyError(f"{key} not found in environment variables.")


__all__ = ["CFG", "logger", "get_env"]
