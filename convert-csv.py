import pandas as pd
from pathlib import Path
from functools import reduce 

path = Path("../streetscapes-data")
file_paths = path.glob('*.csv')

dfs = []
for f in file_paths:
    df = pd.read_csv(f, dtype={'sequence_id': str})
    dfs.append(df)

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=["uuid", "source", "orig_id"],
                                            how='left'), dfs)

df_merged.to_parquet("../streetscapes-data/streetscapes-data.parquet", compression='gzip')