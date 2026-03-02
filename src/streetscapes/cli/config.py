import json

from cyclopts import App
from rich.table import Table

from streetscapes import config
from streetscapes.cli.console import console

config_cli = App(name="config")


@config_cli.command(name="set")
def set_config(key: str, value: str):
    """Set a global streetscapes config value."""
    config.setopt(key, value)
    print(f"Config '{key}' set to '{value}'.")


@config_cli.command(name="get")
def get_config(key: str):
    """Get a config value."""
    value = config.getopt(key)
    if value is not None:
        print(value)
    else:
        print(f"No config value for '{key}'", err=True)
        raise SystemExit(code=1)


@config_cli.command(name="list")
def list_config(
    json_output: bool = False,
):
    """List configuration settings.

    Parameters
    ----------
    json_output:
        Show configuration as JSON if True.
    """
    cfg = config.load()

    if json_output:
        print(json.dumps(cfg, indent=2))
        return

    table = Table(title="Streetscapes Configuration")
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    for key, val in cfg.items():
        table.add_row(key, str(val))

    console.print(table)
