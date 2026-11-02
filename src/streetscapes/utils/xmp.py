"""XMP metadata extraction, used to recognise panoramic images.

Images carry XMP as an XML packet embedded in the file (Adobe's Extensible
Metadata Platform, https://www.adobe.com/devnet/xmp.html). Panorama writers
describe their projection in the `GPano` namespace of Google's spherical image
metadata (https://developers.google.com/streetview/spherical-metadata), which is
the only place a locally imported image says whether it is a panorama; plain
EXIF has no such tag.
"""

from typing import TYPE_CHECKING

from lxml import etree

from streetscapes.utils.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

# warnings quote XMP read from an image file, which can contain
# angle brackets that the colourised logger would take for markup tags
logger = logger.opt(colors=False)

GPANO_NS = "http://ns.google.com/photos/1.0/panorama/"

# An XMP packet is delimited by these tags. The XMP specification allows a bare
# `rdf:RDF` element as well, so both spellings are searched for.
_PACKET_TAGS = (
    (b"<x:xmpmeta", b"</x:xmpmeta>"),
    (b"<rdf:RDF", b"</rdf:RDF>"),
)


def find_xmp_packet(data: bytes) -> bytes | None:
    """Locate the XMP packet embedded in an image file.

    Only the first (main) packet is returned; the extended packets that large
    XMP payloads are split over never carry `GPano` properties.

    Args:
        data: The contents of an image file.

    Returns:
        The XMP packet as XML, or None if the file holds none.
    """
    for open_tag, close_tag in _PACKET_TAGS:
        start = data.find(open_tag)
        if start == -1:
            continue

        end = data.find(close_tag, start)
        if end == -1:
            logger.warning("Unterminated XMP packet; ignoring it.")
            continue

        return data[start : end + len(close_tag)]

    return None


def _local_name(qualified_name: str, namespace: str) -> str | None:
    """Strip the namespace from an element or attribute name.

    Args:
        qualified_name: A name as lxml reports it (`{namespace}name`).
        namespace: The namespace the name is expected to be in.

    Returns:
        The bare name, or None if it belongs to a different namespace.
    """
    prefix = f"{{{namespace}}}"
    if not qualified_name.startswith(prefix):
        return None
    return qualified_name[len(prefix) :]


def parse_gpano_properties(packet: bytes) -> dict[str, str]:
    """Collect the `GPano` properties of an XMP packet.

    Args:
        packet: An XMP packet as XML.

    Returns:
        The properties, keyed by their name without the namespace.
    """
    try:
        parser = etree.XMLParser(
            resolve_entities=False, no_network=True, huge_tree=False
        )
        root = etree.fromstring(packet, parser=parser)
    except etree.ParseError as error:
        logger.warning(f"Malformed XMP packet ({error}); ignoring it.")
        return {}

    properties: dict[str, str] = {}

    for element in root.iter():
        # Comments and processing instructions have a callable for a tag.
        if not isinstance(element.tag, str):
            continue

        for qualified_name, value in element.attrib.items():
            name = _local_name(qualified_name, GPANO_NS)
            if name is not None:
                properties[name] = value

        name = _local_name(element.tag, GPANO_NS)
        if name is not None and element.text is not None and element.text.strip():
            properties[name] = element.text.strip()

    return properties


def _as_bool(value: str) -> bool | None:
    """Interpret an XMP boolean."""
    normalised = value.strip().lower()
    if normalised in {"true", "1", "yes"}:
        return True
    if normalised in {"false", "0", "no"}:
        return False
    logger.warning(f"Uninterpretable XMP boolean '{value}'.")
    return None


def is_panoramic(impath: Path) -> bool | None:
    """Is an image is a panorama, according to its XMP metadata.

    Args:
        impath: Path to an image.

    Returns:
        True or False if the metadata contains project info. None if
        the image carries no valid `GPano` property.
    """
    packet = find_xmp_packet(impath.read_bytes())
    if packet is None:
        return None

    properties = parse_gpano_properties(packet)
    if not properties:
        return None

    projection = properties.get("ProjectionType")
    if projection is not None:
        normalised = projection.strip().lower()
        if normalised in {"equirectangular", "cylindrical", "spherical"}:
            return True
        if normalised in {"rectilinear", "flat", "perspective", "plane"}:
            return False
        logger.warning(f"Unknown XMP projection type '{projection}' for {impath.name}.")

    # A panorama viewer is only ever requested for a panorama.
    viewer = properties.get("UsePanoramaViewer")
    if viewer is not None:
        use_viewer = _as_bool(viewer)
        if use_viewer is not None:
            return use_viewer

    # The dimensions of the full panorama that the image is a part of are
    # required for equirectangular panoramas and absent from flat photographs.
    if "FullPanoWidthPixels" in properties or "FullPanoHeightPixels" in properties:
        return True

    return None
