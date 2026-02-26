from pathlib import Path
from cyclopts import App
from streetscapes import config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ibis

database_cli = App(help="Get info and delete entries from the database.")
segmentations_cli = App(help="View and/or delete segmentation data.")
database_cli.command(segmentations_cli, name="segmentations")


def _get_db() -> "ibis.BaseBackend":
    import ibis

    dbpath = Path(
        f"{config.get("data_home")}/projects/{config.get("active_project")}.duckdb"
    )
    return ibis.duckdb.connect(dbpath, extensions=["spatial", "json"])


@segmentations_cli.command(name="stats")
def segmentation_stats():
    """Get an overview of all segmentation runs in the database."""
    db = _get_db()
    t_segs = db.table("segmentations")
    t_runs = db.table("runs")
    if t_segs.nunique().to_pandas() == 0:
        print(f"The '{config.get("active_project")}' segmentations table is empty")
    else:
        runs = list(t_runs.select("run").to_pandas()["run"])
        print(
            f"Segmentation runs in project '{config.get("active_project")}' database:"
        )
        print(f"{'Name': <37}| Entries")
        print("-"*37 + "+" + "-"*8)
        
        for run in runs:
            n_items = t_segs.filter(t_segs.run==run).nunique().to_pandas()
            print(f"{run: <37}| {n_items: >7}")


@segmentations_cli.command(name="delete")
def delete_segmentations(
    *,
    run_id: str,
):
    """Remove segmentation runs from database

    Args:
        run_id: Run ID that you want to delete. Use '*' to remove all segmention runs.
    """
    db = _get_db()
    t_runs = db.table("runs")

    runs = list(t_runs.select("run").to_pandas()["run"])
    
    if run_id == "*":
        reply = input(
            "This will remove all segmentation runs from the "
            f"'{config.get("active_project")}' project database, are you sure? [y/N]"
        )
        if reply in ["y", "Y"]:
            db.raw_sql("DELETE FROM segmentations WHERE True;")
            db.raw_sql("DELETE FROM runs WHERE True;")
    elif run_id not in runs:
        print(f"Run ID {'run_id'} not found in database")
    else:
        db.raw_sql(f"DELETE FROM segmentations WHERE run='{run_id}';")
        db.raw_sql(f"DELETE FROM runs WHERE run='{run_id}';")
