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
    df = con.table("downloads").head(head).to_pandas()
    if df.empty:
        print("[bold yellow]No entries found in manifest.[/bold yellow]")
        return
    table = df_to_table(df)
    print(table)
