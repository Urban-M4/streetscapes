import typer

from streetscapes import config

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
