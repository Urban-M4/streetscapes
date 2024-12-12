# --------------------------------------
import enum

class SourceMap(enum.Enum):
    '''
    An Enum class representing a source map.
    Currently, only Mapillary and KartaView are supported.
    '''
    Mapillary = enum.auto()
    KartaView = enum.auto()
    GoogleMaps = enum.auto()
