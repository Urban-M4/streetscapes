# --------------------------------------
import pandas as pd

# --------------------------------------
from functools import reduce

# --------------------------------------
from pathlib import Path

# --------------------------------------
from streetscapes import conf


def convert_csv_to_parquet(
    data_dir: Path = conf.DATA_DIR,
    parquet_filename: str = "streetscapes-data.parquet",
):
    """
    Converts a list of global-streetscapes csv files into a single merged dataframe.

    Constructs a pd.DataFrame resulting from merging all csvs on the
    columns "uuid", "source", and "orig_id" using a left join.

    Args:
        data_dir (Path, optional):
            The data directory containing the CSV files.
            Defaults to conf.STREETSCAPES_DATA_DIR.

        parquet_filename (str, optional):
            The name of the Parquet file. It will be saved in the same directory as the CSV files.
            Defaults to "streetscapes-data.parquet"
    """

    if not data_dir.exists():
        raise FileNotFoundError("The data directory does not exist.")

    parquet_file = conf.OUTPUT_DIR / parquet_filename

    if parquet_file.exists():
        ok = input("==[ The target filename exists. Overwrite? (y/[n]) ")
        if not ok.lower().startswith("y"):
            print(f"==[ Exiting.")
            return

    csv_files = data_dir.glob("*.csv")

    csv_dfs = []
    dtypes = {
        "sequence_id": str,
        "capital": str,
        "pano_status": str,
        "view_direction": str,
    }
    for file in csv_files:
        print(f"==[ Processing file {file.relative_to(conf.ROOT_DIR)}...")
        df = pd.read_csv(file, dtype=dtypes)
        df["orig_id"] = df["orig_id"].astype("int64")
        csv_dfs.append(df)

    print(f"==[ Merging files...")
    merged_df = reduce(
        lambda left, right: pd.merge(
            left, right, on=["uuid", "source", "orig_id"], how="left"
        ),
        csv_dfs,
    )

    print(f"==[ Saving file {parquet_file.relative_to(conf.ROOT_DIR)}...")
    merged_df.to_parquet(
        parquet_file,
        compression="gzip",
    )
