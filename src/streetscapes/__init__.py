import os
from dotenv import load_dotenv
from streetscapes.utils import logger
from streetscapes.config import CFG

def get_env(key: str):
    load_dotenv()
    value = os.getenv(key, None)
    if value is not None:
        return value
    raise KeyError(f"{key} not found in environment variables.")
