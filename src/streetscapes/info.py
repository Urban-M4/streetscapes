import duckdb
import pandas as pd
from huggingface_hub import hf_hub_download

# from streetscapes import conf


def bold(text):
    """Display text in bold."""
    return f"\033[1m{text}\033[0m"


def render_info_csv():
    """Display info.csv in a readable form."""

    # (Re)download if not already there
    hf_hub_download(
        repo_id="NUS-UAL/global-streetscapes",
        filename="info.csv",
        repo_type="dataset",
        local_dir=conf.CSV_DIR,
    )

    # Read
    info = pd.read_csv(conf.CSV_DIR / "info.csv")

    # Render
    for _, row in info.iterrows():
        if not pd.isna(row.Filename):
            print(f"- {bold(row.Filename)} - {row.Overview}")
            print(f"  {row.Notes}")
        print(f"    - {bold(row.Field)} ({row.Format}) - {row.Explanation}")


def get_columns(file):
    full_path = conf.CSV_DIR / "data" / file
    return duckdb.sql(f"SELECT * FROM read_csv('{full_path}') LIMIT 0").columns


def get_unique(file, column):
    full_path = conf.CSV_DIR / "data" / file
    return duckdb.sql(f"SELECT DISTINCT {column} FROM read_csv('{full_path}')")
