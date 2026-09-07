"""Time and timestamp utilities."""

from datetime import UTC, datetime


def iso_timestamp(
    precision: str = "seconds",
    fmt: str | None = None,
    sep: str = "T",
    utc: bool = True,
) -> str:
    """Create a date-timestamp as a simplified ISO-formatted string.

    Useful for adding a unique but meaningful string to the
    name of a directory or a file that might be created
    repeatedly with the same name (for instance, when
    running the same experiment multiple times).
    The format is ISO 8601.

    NOTE: UTC time is used to avoid ambiguity.

    Args:
        precision: Precision for the timespec parameter.
        fmt: Explicit format.
        sep: A custom separator for the default ISO format.
        utc: Use UTC time (default)

    Returns:
        The formatted timestamp.
    """
    ts = datetime.now(UTC) if utc else datetime.now()

    if fmt is not None:
        return datetime.strftime(ts, fmt)
    tstr = ts.isoformat(sep=sep, timespec=precision)
    return tstr.split("+")[0]  # remove timezone info
