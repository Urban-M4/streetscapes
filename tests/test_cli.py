import shlex

from typer.testing import CliRunner

from streetscapes.cli.main import app

runner = CliRunner()


def run_cli(cmd: str):
    """Run a CLI command string as if typed in the shell."""
    args = shlex.split(cmd)[1:]  # skip the script name if included
    return runner.invoke(app, args, env={"NO_COLOR": "1"})


class TestCLIHelp:
    """Test the basic structure and help messages of the CLI."""

    def test_main_help(self):
        result = run_cli("streetscapes --help")
        assert result.exit_code == 0
        assert "fetch_metadata" in result.output

    def test_fetch_metadata_help(self):
        result = run_cli("streetscapes fetch_metadata --help")
        assert result.exit_code == 0
        assert "mapillary" in result.output

    def test_fetch_metadata_mapillary_help(self):
        result = run_cli("streetscapes fetch_metadata mapillary --help")
        assert result.exit_code == 0
        assert "--bbox" in result.output
        assert "--tile-size" in result.output
        assert "--limit" in result.output


def test_cli_fetch_metadata_mapillary(fake_mapillary_client, monkeypatch, tmp_path):
    # Replace the real MapillaryClient with the fake one
    monkeypatch.setattr(
        "streetscapes.cli.fetch_metadata._get_mapillary_client",
        lambda token: fake_mapillary_client,
    )

    result = run_cli(f"""
    streetscapes fetch_metadata mapillary \
    --bbox 4.89 52.37 4.91 52.38 \
    --tile-size 0.01 \
    --project {tmp_path / "test_project.duckdb"}
    --token fake_token
    """)

    assert result.exit_code == 0
    assert "Fetching tiles" in result.output
