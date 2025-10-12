import re
import shlex
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from streetscapes.cli.main import app
from streetscapes.project import Project

runner = CliRunner()


def run_cli(cmd: str):
    """Run a CLI command string as if typed in the shell."""
    args = shlex.split(cmd)[1:]  # skip the script name if included
    return runner.invoke(app, args)


def strip_ansi(text: str) -> str:
    """Strip ansi color codes from string.

    This helps to resolve some weird CI issues where e.g. --bbox was
    interspersed with ANSI codes and therefore tests looking for the literal
    text failed.
    """
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)



class TestCLIHelp:
    """Test the basic structure and help messages of the CLI."""

    def test_main_help(self):
        result = run_cli("streetscapes --help")
        assert result.exit_code == 0
        assert "fetch_metadata" in strip_ansi(result.output)

    def test_fetch_metadata_help(self):
        result = run_cli("streetscapes fetch_metadata --help")
        assert result.exit_code == 0
        assert "mapillary" in strip_ansi(result.output)

    def test_fetch_metadata_mapillary_help(self):
        result = run_cli("streetscapes fetch_metadata mapillary --help")
        assert result.exit_code == 0
        assert "--bbox" in strip_ansi(result.output)
        assert "--tile-size" in strip_ansi(result.output)
        assert "--limit" in strip_ansi(result.output)

    def test_download_images_mapillary_help(self):
        result = run_cli("streetscapes download_images mapillary --help")
        assert result.exit_code == 0
        assert "--skip-existing" in strip_ansi(result.output)
        assert "--token" in strip_ansi(result.output)

    def test_export_help(self):
        result = run_cli("streetscapes export --help")
        assert result.exit_code == 0
        assert "table" in strip_ansi(result.output)

    def test_export_table_help(self):
        result = run_cli("streetscapes export table --help")
        assert result.exit_code == 0
        assert "table_name" in strip_ansi(result.output)
        assert "output" in strip_ansi(result.output)


def test_fetch_and_export_integration(fake_mapillary_client, monkeypatch, tmp_path):
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
    assert result.exit_code == 0
    assert "Fetching tiles" in strip_ansi(result.output)

    # -----------------------
    # Test export to CSV
    # -----------------------
    csv_file = tmp_path / "mapillary.csv"
    result_csv = run_cli(f"streetscapes export table mapillary {csv_file}")
    assert result_csv.exit_code == 0
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    # -----------------------
    # Test export to Parquet
    # -----------------------
    parquet_file = tmp_path / "mapillary.parquet"
    result_parquet = run_cli(f"streetscapes export table mapillary {parquet_file}")
    assert result_parquet.exit_code == 0
    assert parquet_file.exists()

    # Read back Parquet to verify content
    project = Project("test_streetscapes")
    df_out = pd.read_parquet(parquet_file)
    table_expr = project.get_table("mapillary")
    count = table_expr.count().execute()
    assert len(df_out) == count
    assert "geometry" in df_out.columns

    # -----------------------
    # Test download CLI (mapillary)
    # -----------------------
    result_dl = run_cli("streetscapes download_images mapillary --skip-existing")
    assert result_dl.exit_code == 0

    # Verify filesystem
    image_dir = project.image_dir("mapillary")
    all_files = list(image_dir.rglob("*.jpg"))
    assert len(all_files) > 0
    for f in all_files:
        assert f.read_text() == "FAKE IMAGE DATA"

    # Verify local_images table
    rows = project.con.raw_sql("SELECT * FROM local_images").fetchall()
    assert len(rows) == len(all_files)
    for row in rows:
        assert Path(row[3]).exists()  # path column points to actual file