import pandas as pd
from pathlib import Path
from functools import reduce 

def convert_csvs_to_dataframe(files):
    """
    Converts a list of global-streetscapes csv files into a single merged dataframe.

    Args:
        files: An iterable of global-streetscapes file paths to csv files.

    Returns:
        pd.DataFrame resulting from merging all csvs on the
        columns "uuid", "source", and "orig_id" using a left join.
    """
    csv_dfs = []
    for f in files:
        df = pd.read_csv(f, dtype={'sequence_id': str})
        csv_dfs.append(df)

    merged_df = reduce(lambda  left,right: pd.merge(left,right,on=["uuid", "source", "orig_id"],
                                            how='left'), csv_dfs)
    return merged_df

if __name__ == "__main__":
    path = Path("../streetscapes-data")
    file_paths = path.glob('*.csv')
    streetscapes_df = convert_csvs_to_dataframe(file_paths)
    streetscapes_df.to_parquet("../streetscapes-data/streetscapes-data.parquet", compression='gzip')