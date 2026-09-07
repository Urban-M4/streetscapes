"""Helpers shared by the segmentation models."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Delay slow imports for CLI responsiveness
    import torch


def get_device(device: "torch.device | str | None") -> "torch.device":
    """Get a Torch device.

    Args:
        device: A string / torch.device specification or None for a sane default.

    Returns:
        A torch.device object.
    """
    import torch

    if isinstance(device, torch.device):
        return device

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.mps.is_available() else "cpu")
        )
    return torch.device(device)


def extract_categories(
    prompt: str | list[str],
    as_list: bool = False,
) -> str | list[str]:
    """Extract labels (object categories) to look for from a free-form prompt.

    Args:
        prompt: The labels as a string or a list of strings.
            If a string is provided, the categories should be
            separated by commas or full stops.
        as_list: Return the prompt as a list of strings rather
            than joining all the strings together into a single prompt.

    Returns:
        A list of labels (object categories).
    """

    def flatten(xs: Iterable):
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x

    if not isinstance(prompt, str):
        prompt = ".".join(flatten(prompt))

    prompt = prompt.strip().lower()

    prompt = ".".join(
        [cat.strip() for cat in prompt.split(",") if len(cat.strip()) > 0]
    )
    prompt = ". ".join(
        [cat.strip() for cat in prompt.split(".") if len(cat.strip()) > 0]
    )

    if as_list:
        return [cat.strip() for cat in prompt.split(".") if len(cat.strip()) > 0]

    return f"{prompt.strip()}."
