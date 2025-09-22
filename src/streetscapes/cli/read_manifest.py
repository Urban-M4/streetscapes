import typer
import ibis
from pathlib import Path

from rich.table import Table
from rich import print

manifest_cli = typer.Typer(help="Preview DuckDB manifest table.")


def df_to_table(df):
    """Convert pandas DataFrame to rich Table."""
    table = Table(show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(str(col))
    for row in df.itertuples(index=False):
        table.add_row(*[str(x) for x in row])

    return table


@manifest_cli.command("head")
def read_manifest(
    manifest_path: Path = typer.Argument(..., help="Path to DuckDB manifest file"),
    head: int = typer.Option(10, help="Number of rows to preview"),
):
    con = ibis.duckdb.connect(str(manifest_path))

    # TODO: generalize this
    if "images" in con.tables:
        df = con.table("images").head(head).to_pandas()
    elif "downloads" in con.tables:
        df = con.table("downloads").head(head).to_pandas()
    else:
        raise ValueError("Unknown manifest format")

    if df.empty:
        print("[bold yellow]No entries found in manifest.[/bold yellow]")
        return
    table = df_to_table(df)
    print(table)


### Some old snippets for reading (geoparquet) manifests in various ways

# def read_manifest(manifest_file: Path):
#     import ibis

#     con = ibis.duckdb.connect()
#     con.load_extension("spatial")
#     return con.read_parquet(manifest_file).to_pandas()

# Alternative (read directly in python)
# return gpd.read_parquet(output_file)

# Alternative with ibis directly
# return ibis.read_parquet(output_file).to_pandas # doesn't handle geometry

# Alternative with geopackage
# return con.read_geo(manifest_file).to_pandas()