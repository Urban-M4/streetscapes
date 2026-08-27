"""Finetune CLI (not implemented)."""

import typer

finetune_model_cli = typer.Typer(help="Finetune segmentation models.")


@finetune_model_cli.command("dinosam")
def finetune_dinosam(
    train: str = typer.Option(..., help="Path to training data"),
    validate: str = typer.Option(..., help="Path to validation data"),
    test: str = typer.Option(..., help="Path to test data"),
):
    """Finetune the DinoSAM model."""
    # Placeholder for actual finetuning logic
    typer.echo(f"Training DinoSAM with train={train}, validate={validate}, test={test}")
    raise NotImplementedError("DinoSAM finetuning not yet implemented")


# ... and likewise for other models
