from pprint import pp

from cyclopts import App
from rich.table import Table

from streetscapes import CFG
from streetscapes.cli.console import console

config_cli = App(name="config")


@config_cli.command(name="set")
def set_config(key: str, value: str):
    """
    Set a global streetscapes config value

    Args:
        key: The configuration option.
        value: The value to set the option to.

    Raises:
        SystemExit: Raised if the configuration option does not exist.
    """
    from streetscapes.project import Project

    if not hasattr(CFG, key):
        print(f"No config value for '{key}'", err=True)
        raise SystemExit(code=1)

    # For now, this cannot be done in a Pydantic validator
    # because it triggers a circular import error.
    if key == "active_project":
        proj = Project(value)

    setattr(CFG, key, value)
    CFG.save()
    print(f"Config '{key}' set to '{value}'.")


@config_cli.command(name="get")
def get_config(key: str):
    """
    Get a config value.

    Args:
        key: The configuration option.

    Raises:
        SystemExit: Raised if the configuration option does not exist.
    """

    value = getattr(CFG, key)
    if value is not None:
        print(value)
    else:
        print(f"No config value for '{key}'", err=True)
        raise SystemExit(code=1)


@config_cli.command(name="list")
def list_config(
    json_output: bool = False,
    indent: int = 2,
):
    """List configuration settings.

    Args:
        json_output: Show configuration as JSON if True.
        indent: Indentation for JSON output.
    """

    if json_output:
        pp(CFG.model_dump_json(indent=indent))
        return

    cfg = CFG.model_dump(mode="python")

    table = Table(title="Streetscapes Configuration")
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    for key, val in cfg.items():
        table.add_row(key, str(val))

    console.print(table)
