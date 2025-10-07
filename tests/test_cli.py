import re
import shlex

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

    def test_export_help(self):
        result = run_cli("streetscapes export --help")
        assert result.exit_code == 0
        assert "table" in strip_ansi(result.output)

    def test_export_table_help(self):
        result = run_cli("streetscapes export table --help")
        assert result.exit_code == 0
        assert "table_name" in strip_ansi(result.output)
        assert "output" in strip_ansi(result.output)
        assert "--project" in strip_ansi(result.output)


def test_fetch_and_export_integration(fake_mapillary_client, monkeypatch, tmp_path):
    # -----------------------
    # 1. Fetch Mapillary metadata via CLI
    # -----------------------
    monkeypatch.setattr(
        "streetscapes.cli.fetch_metadata._get_mapillary_client",
        lambda token: fake_mapillary_client,
    )

    project_file = tmp_path / "test_project.duckdb"

    fetch_cmd = f"""
    streetscapes fetch_metadata mapillary \
    --bbox 4.89 52.37 4.91 52.38 \
    --tile-size 0.01 \
    --project {project_file} \
    --token fake_token
    """
    result = run_cli(fetch_cmd)
    assert result.exit_code == 0
    assert "Fetching tiles" in strip_ansi(result.output)

    # -----------------------
    # 2. Export to CSV
    # -----------------------
    csv_file = tmp_path / "mapillary.csv"
    export_csv_cmd = (
        f"streetscapes export table mapillary {csv_file} --project {project_file}"
    )
    result_csv = run_cli(export_csv_cmd)
    assert result_csv.exit_code == 0
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    # -----------------------
    # 3. Export to Parquet
    # -----------------------
    parquet_file = tmp_path / "mapillary.parquet"
    export_parquet_cmd = (
        f"streetscapes export table mapillary {parquet_file} --project {project_file}"
    )
    result_parquet = run_cli(export_parquet_cmd)
    assert result_parquet.exit_code == 0
    assert parquet_file.exists()

    # Optionally, read back Parquet to verify content
    project = Project(db_path=str(project_file))
    df_out = pd.read_parquet(parquet_file)
    table_expr = project.get_table("mapillary")
    count = table_expr.count().execute()
    assert len(df_out) == count
    assert "geometry" in df_out.columns
