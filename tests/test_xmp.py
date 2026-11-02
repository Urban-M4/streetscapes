from typing import TYPE_CHECKING

import pytest
from PIL import Image

from streetscapes.utils.xmp import (
    GPANO_NS,
    find_xmp_packet,
    is_panoramic,
    parse_gpano_properties,
)

if TYPE_CHECKING:
    from pathlib import Path

RDF_OPEN = '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
META_OPEN = '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
CLOSE = "</rdf:RDF></x:xmpmeta>"

# The wrapping that an XMP packet is embedded in an image file with.
XPACKET_HEADER = b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
XPACKET_TRAILER = b'<?xpacket end="w"?>'


def packet(description: str) -> bytes:
    """Wrap an `rdf:Description` element in a complete XMP packet."""
    return f"{META_OPEN}{RDF_OPEN}{description}{CLOSE}".encode()


def attribute_packet(**properties: str) -> bytes:
    """Build an XMP packet holding `GPano` properties as attributes."""
    attributes = " ".join(f'GPano:{k}="{v}"' for k, v in properties.items())
    return packet(
        f'<rdf:Description rdf:about="" xmlns:GPano="{GPANO_NS}" {attributes}/>'
    )


def element_packet(**properties: str) -> bytes:
    """Build an XMP packet holding `GPano` properties as child elements."""
    elements = "".join(f"<GPano:{k}>{v}</GPano:{k}>" for k, v in properties.items())
    return packet(
        f'<rdf:Description rdf:about="" xmlns:GPano="{GPANO_NS}">'
        f"{elements}</rdf:Description>"
    )


@pytest.fixture
def write_image(tmp_path):
    """Write a JPEG, optionally with an XMP packet embedded in it."""

    def _write(name: str, xmp: bytes | None = None) -> Path:
        path = tmp_path / name
        image = Image.new("RGB", (64, 32))
        if xmp is None:
            image.save(path)
        else:
            image.save(path, xmp=XPACKET_HEADER + xmp + XPACKET_TRAILER)
        return path

    return _write


class TestFindXmpPacket:
    def test_finds_the_packet_in_a_jpeg(self, write_image):
        path = write_image("pano.jpg", attribute_packet(ProjectionType="spherical"))

        found = find_xmp_packet(path.read_bytes())

        assert found is not None
        # The xpacket wrapping is not part of the XML document.
        assert found.startswith(b"<x:xmpmeta")
        assert found.endswith(b"</x:xmpmeta>")

    def test_no_xmp_gives_none(self, write_image):
        assert find_xmp_packet(write_image("flat.jpg").read_bytes()) is None

    def test_accepts_a_bare_rdf_packet(self):
        """Writers are allowed to leave out the `x:xmpmeta` wrapper."""
        bare = f"{RDF_OPEN}</rdf:RDF>".encode()

        assert find_xmp_packet(b"\xff\xd8junk" + bare + b"junk") == bare

    def test_unterminated_packet_is_ignored(self):
        assert find_xmp_packet(META_OPEN.encode()) is None


class TestParseGpanoProperties:
    def test_reads_attributes(self):
        properties = parse_gpano_properties(
            attribute_packet(
                ProjectionType="equirectangular", FullPanoWidthPixels="8000"
            )
        )

        assert properties == {
            "ProjectionType": "equirectangular",
            "FullPanoWidthPixels": "8000",
        }

    def test_reads_child_elements(self):
        properties = parse_gpano_properties(
            element_packet(ProjectionType="equirectangular")
        )

        assert properties == {"ProjectionType": "equirectangular"}

    def test_ignores_other_namespaces(self):
        other = packet(
            '<rdf:Description rdf:about="" xmlns:tiff="http://ns.adobe.com/tiff/1.0/"'
            ' tiff:ProjectionType="equirectangular"/>'
        )

        assert parse_gpano_properties(other) == {}

    def test_prefix_may_be_renamed(self):
        """Only the namespace identifies a property; the prefix is arbitrary."""
        renamed = packet(
            f'<rdf:Description rdf:about="" xmlns:sphere="{GPANO_NS}"'
            ' sphere:ProjectionType="equirectangular"/>'
        )

        assert parse_gpano_properties(renamed) == {"ProjectionType": "equirectangular"}

    def test_malformed_packet_gives_no_properties(self):
        assert parse_gpano_properties(b"<x:xmpmeta") == {}

    def test_entities_are_not_expanded(self):
        """A packet from an image file is untrusted; entity bombs must not blow up."""
        bomb = (
            "<!DOCTYPE x [<!ENTITY a 'boom'>"
            "<!ENTITY b '&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;'>"
            "<!ENTITY c '&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;'>]>"
            + META_OPEN
            + RDF_OPEN
            + f'<rdf:Description rdf:about="" xmlns:GPano="{GPANO_NS}">'
            "<GPano:ProjectionType>&c;</GPano:ProjectionType>"
            "<GPano:PoseHeadingDegrees>12</GPano:PoseHeadingDegrees>"
            "</rdf:Description>" + CLOSE
        ).encode()

        # The rest of the packet still parses; the unresolved entity is dropped.
        assert parse_gpano_properties(bomb) == {"PoseHeadingDegrees": "12"}

    def test_reads_a_real_packet(self):
        """Verbatim XMP from a Hugin-stitched panorama on Wikimedia Commons.

        Panorama writers turn out to favour the child-element form, and to spread
        their properties over several `rdf:Description` elements, so keep a real
        packet around with exactly the quirks that a synthetic one smooths over:
        single-quoted attributes, newlines around the values, a second
        description in another namespace, and an `x:xmptk` attribute.
        """
        real = b"""<x:xmpmeta xmlns:x='adobe:ns:meta/' x:xmptk='Image::ExifTool 11.16'>
<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>

 <rdf:Description rdf:about=''
  xmlns:GPano='http://ns.google.com/photos/1.0/panorama/'>
  <GPano:CroppedAreaImageHeightPixels>5000</GPano:CroppedAreaImageHeightPixels>
  <GPano:CroppedAreaImageWidthPixels>10000</GPano:CroppedAreaImageWidthPixels>
  <GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>
  <GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>
  <GPano:FullPanoHeightPixels>5000</GPano:FullPanoHeightPixels>
  <GPano:FullPanoWidthPixels>10000</GPano:FullPanoWidthPixels>
  <GPano:ProjectionType>equirectangular</GPano:ProjectionType>
  <GPano:SourcePhotosCount>5</GPano:SourcePhotosCount>
  <GPano:StitchingSoftware>Hugin</GPano:StitchingSoftware>
  <GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>
 </rdf:Description>

 <rdf:Description rdf:about=''
  xmlns:exif='http://ns.adobe.com/exif/1.0/'>
  <exif:GPSLatitude>32,53.26608N</exif:GPSLatitude>
  <exif:GPSLongitude>117,13.72836W</exif:GPSLongitude>
 </rdf:Description>
</rdf:RDF>
</x:xmpmeta>"""

        properties = parse_gpano_properties(real)

        assert properties == {
            "CroppedAreaImageHeightPixels": "5000",
            "CroppedAreaImageWidthPixels": "10000",
            "CroppedAreaLeftPixels": "0",
            "CroppedAreaTopPixels": "0",
            "FullPanoHeightPixels": "5000",
            "FullPanoWidthPixels": "10000",
            "ProjectionType": "equirectangular",
            "SourcePhotosCount": "5",
            "StitchingSoftware": "Hugin",
            "UsePanoramaViewer": "True",
        }

    def test_ignores_a_packet_full_of_other_namespaces(self):
        """Processed photographs carry sizeable XMP without a single `GPano` tag."""
        edited = packet(
            '<rdf:Description rdf:about=""'
            ' xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"'
            ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
            ' xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"'
            ' crs:Version="7.0" photoshop:City="Cuxhaven">'
            "<dc:subject><rdf:Bag><rdf:li>street</rdf:li></rdf:Bag></dc:subject>"
            "</rdf:Description>"
        )

        assert parse_gpano_properties(edited) == {}


class TestIsPanoramic:
    @pytest.mark.parametrize(
        "projection", ["equirectangular", "cylindrical", "spherical", "EquiRectangular"]
    )
    def test_panoramic_projections(self, write_image, projection):
        path = write_image("pano.jpg", attribute_packet(ProjectionType=projection))

        assert is_panoramic(path) is True

    @pytest.mark.parametrize("projection", ["rectilinear", "flat", "perspective"])
    def test_flat_projections(self, write_image, projection):
        path = write_image("flat.jpg", attribute_packet(ProjectionType=projection))

        assert is_panoramic(path) is False

    def test_no_xmp_is_undecided(self, write_image):
        assert is_panoramic(write_image("plain.jpg")) is None

    def test_xmp_without_gpano_is_undecided(self, write_image):
        path = write_image(
            "plain.jpg",
            packet(
                '<rdf:Description rdf:about=""'
                ' xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Make="Acme"/>'
            ),
        )

        assert is_panoramic(path) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("True", True), ("true", True), ("1", True), ("False", False), ("0", False)],
    )
    def test_falls_back_to_the_panorama_viewer_flag(self, write_image, value, expected):
        """Without a projection type, asking for a panorama viewer settles it."""
        path = write_image("pano.jpg", attribute_packet(UsePanoramaViewer=value))

        assert is_panoramic(path) is expected

    def test_falls_back_to_the_full_panorama_dimensions(self, write_image):
        """Only a panorama reports the dimensions of the sphere it is cropped from."""
        path = write_image(
            "pano.jpg",
            attribute_packet(FullPanoWidthPixels="8000", FullPanoHeightPixels="4000"),
        )

        assert is_panoramic(path) is True

    def test_projection_wins_over_the_viewer_flag(self, write_image):
        path = write_image(
            "flat.jpg",
            attribute_packet(ProjectionType="rectilinear", UsePanoramaViewer="True"),
        )

        assert is_panoramic(path) is False

    def test_unknown_projection_falls_through(self, write_image):
        path = write_image(
            "pano.jpg",
            attribute_packet(ProjectionType="fisheye", FullPanoWidthPixels="8000"),
        )

        assert is_panoramic(path) is True

    def test_unknown_projection_alone_is_undecided(self, write_image):
        path = write_image("mystery.jpg", attribute_packet(ProjectionType="fisheye"))

        assert is_panoramic(path) is None
