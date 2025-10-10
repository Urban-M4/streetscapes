import json

import typer
from rich.table import Table

from streetscapes import config
from streetscapes.cli import console

config_cli = typer.Typer(name="config")


@config_cli.command("set")
def set_config(key: str, value: str):
    """Set a global streetscapes config value."""
    config.set(key, value)
    typer.echo(f"Config '{key}' set to '{value}'.")


@config_cli.command("get")
def get_config(key: str):
    """Get a config value."""
    value = config.get(key)
    if value is not None:
        typer.echo(value)
    else:
        typer.echo(f"No config value for '{key}'", err=True)
        raise typer.Exit(code=1)


@config_cli.command("list")
def list_config(json_output: bool = typer.Option(False, "--json", help="Show as JSON")):
    """List all configuration values."""
    cfg = config.load()

    if json_output:
        typer.echo(json.dumps(cfg, indent=2))
        return

    table = Table(title="Streetscapes Configuration")
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    for key, val in cfg.items():
        table.add_row(key, str(val))

    console.print(table)