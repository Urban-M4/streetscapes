import enum


class CIEnum(enum.StrEnum):
    """
    Case-insensitive string enum.
    """

    @classmethod
    def _missing_(cls, value: str):
        for member in cls:
            if (
                member.name.replace("_", "").casefold()
                == value.replace("_", "").casefold()
            ):
                return member


class Attr(CIEnum):
    """
    Instance attributes whose statistics can be computed.
    """

    R = enum.auto()
    G = enum.auto()
    B = enum.auto()
    H = enum.auto()
    S = enum.auto()
    V = enum.auto()
    Area = enum.auto()
    Coords = enum.auto()

    @classmethod
    @property
    def RGB(cls):
        return {Attr.R, Attr.G, Attr.B}

    @classmethod
    @property
    def HSV(cls):
        return {Attr.H, Attr.S, Attr.V}

    @classmethod
    @property
    def JSONColour(cls):
        return [a.name.lower() for a in Attr.HSV.union(Attr.RGB)]


class Stat(CIEnum):
    """
    Types of statistics.
    """

    Mean = enum.auto()
    Mode = enum.auto()
    Median = enum.auto()
    SD = enum.auto()
