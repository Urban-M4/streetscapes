import io
import re
import shlex
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import pytest

from streetscapes.cli.main import app
from streetscapes.project import Project


def run_cli(cmd: str, exit_code=0):
    """
    Run a Cyclopts CLI command string as if typed in the shell.

    Args:
        cmd: Command line string, e.g., "streetscapes config list --json"

    Returns:
        Captured standard output.
    """
    args = shlex.split(cmd)[1:]  # skip the script name if present

    stdout = io.StringIO()

    with redirect_stdout(stdout), pytest.raises(SystemExit) as sysexit:
        app(args)

    assert sysexit.value.code == exit_code
    return stdout.getvalue()



class TestCLIHelp:
    """Test the basic structure and help messages of the CLI."""

    def test_main_help(self):
        result = run_cli("streetscapes --help")
        assert "fetch-metadata" in result

    def test_fetch_metadata_help(self):
        result = run_cli("streetscapes fetch-metadata --help")
        assert "mapillary" in result

    def test_fetch_metadata_mapillary_help(self):
        result = run_cli("streetscapes fetch-metadata mapillary --help")
        assert "BBOX" in result
        assert "--tile-size" in result
        assert "--limit" in result

    def test_download_images_mapillary_help(self):
        result = run_cli("streetscapes download-images mapillary --help")
        assert "--skip-existing" in result
        assert "--token" in result

    def test_export_help(self):
        result = run_cli("streetscapes export --help")
        assert "table" in result

    def test_export_table_help(self):
        result = run_cli("streetscapes export table --help")
        assert "TABLE_NAME" in result
        assert "OUTPUT" in result


@pytest.mark.skip(reason="TODO use memisis data instead of fake_mapillary_data")
def test_fetch_and_export_integration(fake_mapillary_client, tmp_path):
    # -----------------------
    # Print test config
    # -----------------------
    result_cfg = run_cli("streetscapes config get active_project")
    assert "test_streetscapes" in result_cfg

    # -----------------------
    # Fetch Mapillary metadata via CLI
    # -----------------------
    fetch_cmd = """
    streetscapes fetch_metadata mapillary \
    --bbox 4.89 52.37 4.91 52.38 \
    --tile-size 0.01 \
    --token fake_token
    """
    result = run_cli(fetch_cmd)
    assert "Fetching tiles" in result

    # -----------------------
    # Test export to CSV
    # -----------------------
    csv_file = tmp_path / "mapillary.csv"
    result_csv = run_cli(f"streetscapes export table mapillary {csv_file}")
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    # -----------------------
    # Test export to Parquet
    # -----------------------
    parquet_file = tmp_path / "mapillary.parquet"
    result_parquet = run_cli(f"streetscapes export table mapillary {parquet_file}")
    assert parquet_file.exists()

    # Read back Parquet to verify content
    project = Project("test_streetscapes")
    df_out = pd.read_parquet(parquet_file)
    table_expr = project.ensure_table("mapillary")
    count = table_expr.count().execute()
    assert len(df_out) == count
    assert "geometry" in df_out.columns

    # -----------------------
    # Test download CLI (mapillary)
    # -----------------------
    result_dl = run_cli("streetscapes download-images mapillary --skip-existing")

    # Verify filesystem
    image_dir = project.get_image_dir("mapillary")
    all_files = list(image_dir.rglob("*.jpg"))
    assert len(all_files) > 0
    for f in all_files:
        assert f.read_text() == "FAKE IMAGE DATA"

    # Verify local_images table
    rows = project._con.raw_sql("SELECT * FROM local_images").fetchall()
    assert len(rows) == len(all_files)
    for row in rows:
        assert Path(row[3]).exists()  # path column points to actual file
