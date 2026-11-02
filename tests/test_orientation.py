import numpy as np
import pytest
from PIL import Image

from streetscapes.utils.images import copy_upright, upright_metadata

# Orientation -> the transform a reader must apply to display the image upright.
# 4032x3024 landscape pixels tagged 6 are meant to be shown as 3024x4032.
ORIENTATIONS = {
    1: "as stored",
    2: "mirrored",
    3: "turned half way",
    4: "mirrored and turned half way",
    5: "transposed",
    6: "turned a quarter clockwise",
    7: "transverse",
    8: "turned a quarter anticlockwise",
}


@pytest.fixture
def write_tagged(tmp_path):
    """Write a landscape JPEG carrying the given EXIF orientation."""

    def _write(orientation: int | None, name: str = "img.jpg", size=(64, 32)):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # A gradient, so that a rotation is visible in the pixels themselves.
        pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        pixels[:, :, 0] = np.linspace(0, 255, size[0], dtype=np.uint8)
        pixels[:, :, 1] = np.linspace(0, 255, size[1], dtype=np.uint8)[:, None]
        image = Image.fromarray(pixels)
        exif = image.getexif()
        if orientation is not None:
            exif[0x0112] = orientation
        # Cameras record the stored dimensions in the Exif sub-IFD.
        sub_ifd = exif.get_ifd(0x8769)
        sub_ifd[0xA002], sub_ifd[0xA003] = size
        image.save(path, exif=exif, quality=95)
        return path

    return _write


class TestCopyUpright:
    @pytest.mark.parametrize("orientation", [5, 6, 7, 8])
    def test_a_quarter_turn_swaps_the_axes(self, write_tagged, tmp_path, orientation):
        source = write_tagged(orientation)

        assert copy_upright(source, tmp_path / "out.jpg") is True
        assert Image.open(tmp_path / "out.jpg").size == (32, 64)

    @pytest.mark.parametrize("orientation", [2, 3, 4])
    def test_a_flip_or_half_turn_keeps_the_axes(
        self, write_tagged, tmp_path, orientation
    ):
        source = write_tagged(orientation)

        assert copy_upright(source, tmp_path / "out.jpg") is True
        assert Image.open(tmp_path / "out.jpg").size == (64, 32)

    def test_the_orientation_tag_is_cleared(self, write_tagged, tmp_path):
        """Left in place, the tag would tell readers to turn the image twice."""
        source = write_tagged(6)

        copy_upright(source, tmp_path / "out.jpg")

        assert Image.open(tmp_path / "out.jpg").getexif().get(0x0112) is None

    @pytest.mark.parametrize(
        ("orientation", "expected"), [(6, (32, 64)), (3, (64, 32))]
    )
    def test_the_recorded_dimensions_follow_the_pixels(
        self, write_tagged, tmp_path, orientation, expected
    ):
        """`exif_transpose` leaves PixelX/YDimension describing the old layout."""
        source = write_tagged(orientation)

        copy_upright(source, tmp_path / "out.jpg")

        sub_ifd = Image.open(tmp_path / "out.jpg").getexif().get_ifd(0x8769)
        assert (sub_ifd[0xA002], sub_ifd[0xA003]) == expected

    def test_the_xmp_orientation_is_cleared(self, write_tagged, tmp_path):
        """XMP can repeat the tag as `tiff:Orientation`; it must go as well."""
        from streetscapes.utils.xmp import find_xmp_packet

        source = write_tagged(6)
        image = Image.open(source)
        exif = image.getexif()
        packet = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""'
            b' xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/>'
            b"</rdf:RDF></x:xmpmeta>"
        )
        image.save(source, exif=exif, xmp=packet, quality=95)

        copy_upright(source, tmp_path / "out.jpg")

        assert b"Orientation" not in find_xmp_packet(
            (tmp_path / "out.jpg").read_bytes()
        )

    def test_the_pixels_are_actually_turned(self, write_tagged, tmp_path):
        """A quarter turn clockwise puts the left-hand column along the top."""
        source = write_tagged(6)
        expected = Image.open(source).transpose(Image.Transpose.ROTATE_270)

        copy_upright(source, tmp_path / "out.jpg")

        got = np.asarray(Image.open(tmp_path / "out.jpg"), dtype=np.int16)
        assert np.abs(got - np.asarray(expected, dtype=np.int16)).mean() < 2

    @pytest.mark.parametrize("orientation", [None, 1])
    def test_upright_images_are_copied_verbatim(
        self, write_tagged, tmp_path, orientation
    ):
        """Nothing to do means no re-encode, so the bytes must be untouched."""
        source = write_tagged(orientation)

        assert copy_upright(source, tmp_path / "out.jpg") is False
        assert (tmp_path / "out.jpg").read_bytes() == source.read_bytes()

    @pytest.mark.parametrize("orientation", [0, 9, 42])
    def test_undefined_orientations_are_left_alone(
        self, write_tagged, tmp_path, orientation
    ):
        """0 turns up in the wild; the specification only defines 1-8."""
        source = write_tagged(orientation)

        assert copy_upright(source, tmp_path / "out.jpg") is False
        assert (tmp_path / "out.jpg").read_bytes() == source.read_bytes()

    def test_the_source_is_never_modified(self, write_tagged, tmp_path):
        source = write_tagged(6)
        before = source.read_bytes()

        copy_upright(source, tmp_path / "out.jpg")

        assert source.read_bytes() == before

    def test_other_exif_tags_survive(self, write_tagged, tmp_path):
        source = write_tagged(6)
        image = Image.open(source)
        exif = image.getexif()
        exif[0x010F] = "Acme"  # Make
        exif[0x0110] = "Cam 1"  # Model
        exif[0x0112] = 6
        image.save(source, exif=exif, quality=95)

        copy_upright(source, tmp_path / "out.jpg")

        kept = Image.open(tmp_path / "out.jpg").getexif()
        assert (kept.get(0x010F), kept.get(0x0110)) == ("Acme", "Cam 1")

    def test_xmp_survives(self, write_tagged, tmp_path):
        """A rotated panorama must keep the `GPano` properties it is known by."""
        from streetscapes.utils.xmp import find_xmp_packet, parse_gpano_properties

        source = write_tagged(6)
        image = Image.open(source)
        exif = image.getexif()
        exif[0x0112] = 6
        packet = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""'
            b' xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"'
            b' GPano:ProjectionType="equirectangular"/>'
            b"</rdf:RDF></x:xmpmeta>"
        )
        image.save(source, exif=exif, xmp=packet, quality=95)

        copy_upright(source, tmp_path / "out.jpg")

        found = find_xmp_packet((tmp_path / "out.jpg").read_bytes())
        assert found is not None
        assert parse_gpano_properties(found) == {"ProjectionType": "equirectangular"}

    def test_a_png_is_handled(self, tmp_path):
        """The JPEG-only encoder options must not be reached for other formats."""
        source = tmp_path / "in.png"
        image = Image.new("RGB", (64, 32), "red")
        exif = image.getexif()
        exif[0x0112] = 6
        image.save(source, exif=exif)

        assert copy_upright(source, tmp_path / "out.png") is True
        assert Image.open(tmp_path / "out.png").size == (32, 64)


class TestUprightMetadata:
    @pytest.mark.parametrize("orientation", [5, 6, 7, 8])
    def test_a_quarter_turn_swaps_the_recorded_size(self, orientation):
        metadata = {"orientation": orientation, "width": 4032, "height": 3024}

        upright_metadata(metadata)

        assert metadata == {"orientation": 1, "width": 3024, "height": 4032}

    @pytest.mark.parametrize("orientation", [2, 3, 4])
    def test_a_flip_keeps_the_recorded_size(self, orientation):
        metadata = {"orientation": orientation, "width": 4032, "height": 3024}

        upright_metadata(metadata)

        assert metadata == {"orientation": 1, "width": 4032, "height": 3024}

    @pytest.mark.parametrize("orientation", [None, 0, 9])
    def test_undefined_orientations_are_left_alone(self, orientation):
        """`copy_upright` leaves these files untouched, so the metadata must match."""
        metadata = {"orientation": orientation, "width": 4032, "height": 3024}

        upright_metadata(metadata)

        assert metadata == {"orientation": orientation, "width": 4032, "height": 3024}

    def test_a_missing_size_is_not_invented(self):
        """EXIF often omits the dimensions; swapping None for None is harmless."""
        metadata = {"orientation": 6, "width": None, "height": None}

        upright_metadata(metadata)

        assert metadata == {"orientation": 1, "width": None, "height": None}


class TestImportRotation:
    """`add_local_images` end to end, since the stored file and the `local` row
    have to agree with each other."""

    @pytest.fixture
    def project(self, tmp_path):
        from streetscapes.project import Project

        return Project(
            name="orientation", image_dir=tmp_path / "images", project_dir=tmp_path
        )

    def test_the_stored_copy_and_its_row_agree(self, project, write_tagged, tmp_path):
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        write_tagged(6, name="incoming/street.jpg", size=(64, 32))

        project.add_local_images(incoming)

        stored = next((tmp_path / "images").rglob("*.jpg"))
        row = project.table("local").to_pandas().iloc[0]
        assert Image.open(stored).size == (32, 64)
        assert Image.open(stored).getexif().get(0x0112) is None
        assert (row["orientation"], row["width"], row["height"]) == (1, 32, 64)

    def test_switching_it_off_stores_the_bytes_as_they_came(
        self, project, write_tagged, tmp_path
    ):
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        source = write_tagged(6, name="incoming/street.jpg", size=(64, 32))

        project.add_local_images(incoming, auto_rotate=False)

        stored = next((tmp_path / "images").rglob("*.jpg"))
        row = project.table("local").to_pandas().iloc[0]
        assert stored.read_bytes() == source.read_bytes()
        # The reader is left to act on the tag, so the row keeps reporting it.
        assert row["orientation"] == 6

    def test_reimporting_the_same_source_is_idempotent(
        self, project, write_tagged, tmp_path
    ):
        """Turning an image upright is deterministic, so a rotated import dedupes."""
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        write_tagged(6, name="incoming/street.jpg", size=(64, 32))

        project.add_local_images(incoming)
        project.add_local_images(incoming)

        assert len(project.table("images").to_pandas()) == 1
        assert len(list((tmp_path / "images").rglob("*.jpg"))) == 1

    @pytest.mark.parametrize("orientation", [1, 6])
    def test_the_stored_copy_hashes_to_its_uuid(
        self, project, write_tagged, tmp_path, orientation
    ):
        """Whoever re-hashes a stored copy must find it under the same UUID."""
        from streetscapes.utils import get_image_uuid

        incoming = tmp_path / "incoming"
        incoming.mkdir()
        write_tagged(orientation, name="incoming/street.jpg", size=(64, 32))

        project.add_local_images(incoming)

        (stored,) = (tmp_path / "images").rglob("*.jpg")
        row = project.table("images").to_pandas().iloc[0]
        assert get_image_uuid(stored) == row["uuid"]
        assert stored.stem == str(row["uuid"])

    def test_no_temporary_files_are_left_behind(self, project, write_tagged, tmp_path):
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        write_tagged(6, name="incoming/street.jpg", size=(64, 32))

        project.add_local_images(incoming)
        project.add_local_images(incoming)  # skipped as a duplicate this time

        assert len([f for f in (tmp_path / "images").rglob("*") if f.is_file()]) == 1
