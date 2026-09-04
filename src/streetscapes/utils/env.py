"""Environment variable access."""

import os

from dotenv import load_dotenv


def get_env(key: str):
    """Read the value of `key` from the environment variables."""
    load_dotenv()
    value = os.getenv(key, None)

    if value is not None:
        return value

    raise KeyError(f"{key} not found in environment variables.")
