def test_reproducible_uuids():
    """Test if images with different metadata produce the same hash / UUID"""
    import piexif as exif
    import numpy as np
    from streetscapes.utils import functions as F
    from PIL import Image
    import io

    # Create some EXIF data.
    # Example based on the PiExif documentation.
    zeroth_ifd = {
        exif.ImageIFD.Make: "Canon",
        exif.ImageIFD.XResolution: (100, 1),
        exif.ImageIFD.YResolution: (100, 1),
        exif.ImageIFD.Software: "piexif",
    }
    exif_ifd = {
        exif.ExifIFD.DateTimeOriginal: "2099:09:29 10:10:10",
        exif.ExifIFD.LensMake: "LensMake",
        exif.ExifIFD.Sharpness: 65535,
        exif.ExifIFD.LensSpecification: ((1, 1), (1, 1), (1, 1), (1, 1)),
    }
    gps_ifd = {
        exif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
        exif.GPSIFD.GPSAltitudeRef: 1,
        exif.GPSIFD.GPSDateStamp: "1999:99:99 99:99:99",
    }
    first_ifd = {
        exif.ImageIFD.Make: "Canon",
        exif.ImageIFD.XResolution: (100, 1),
        exif.ImageIFD.YResolution: (100, 1),
        exif.ImageIFD.Software: "piexif",
    }

    exif_dict = {
        "0th": zeroth_ifd,
        "Exif": exif_ifd,
        "GPS": gps_ifd,
        "1st": first_ifd,
    }
    exif_bytes = exif.dump(exif_dict)

    # Create a random 100x100 NumPy array
    rand_arr = (np.random.rand(100, 100) * 255).astype(np.ubyte)

    # Save the image in JPEG format into a BytesIO object, without the EXIF data.
    bio_orig = io.BytesIO()
    img = Image.fromarray(rand_arr)
    img.save(bio_orig, "jpeg")

    h_orig = F.get_image_hash(bio_orig).hex()

    # Save the image in JPEG format into a BytesIO object, without the EXIF data.
    bio_exif = io.BytesIO()
    img.save(bio_exif, "jpeg", exif=exif_bytes)

    h_exif = F.get_image_hash(bio_exif).hex()

    assert h_orig == h_exif
