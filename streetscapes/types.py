# --------------------------------------
import typing as tp

# --------------------------------------
import enum

# --------------------------------------
import pandas as pd


class SourceMap(enum.Enum):
    Mapillary = enum.auto()
    KartaView = enum.auto()
    GoogleMaps = enum.auto()
