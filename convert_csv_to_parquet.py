from typing import Iterable
import pandas as pd
from pathlib import Path
from functools import reduce 

streetscapes_path = Path("../streetscapes-data")

def convert_csvs_to_dataframe(files: Iterable[Path]) -> pd.DataFrame:
    """
    Converts a list of global-streetscapes csv files into a single merged dataframe.

    Args:
        files: An iterable of global-streetscapes file paths to csv files.

    Returns:
        pd.DataFrame resulting from merging all csvs on the
        columns "uuid", "source", and "orig_id" using a left join.
    """
    csv_dfs = []
    dtypes = {'sequence_id': str, 'capital': str, 'pano_status': str, 'view_direction': str}
    for file in files:
        df = pd.read_csv(file, dtype=dtypes)
        csv_dfs.append(df)

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=["uuid", "source", "orig_id"], how='left'),
        csv_dfs,
    )
    return merged_df

if __name__ == "__main__":
    file_paths = streetscapes_path.glob('*.csv')
    streetscapes_df = convert_csvs_to_dataframe(file_paths)
    streetscapes_df.to_parquet("../streetscapes-data/streetscapes-data.parquet", compression='gzip')