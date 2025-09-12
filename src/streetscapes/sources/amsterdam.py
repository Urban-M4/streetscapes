# --------------------------------------
from pathlib import Path

# --------------------------------------
import ibis

# --------------------------------------
from streetscapes.sources.base import ImageSourceBase


class AmsterdamPanorama(ImageSourceBase):
    """TODO: Add docstrings"""

    def __init__(
        self,
        root_dir: str | Path | None = None,
    ):
        """
        An interface for downloading and manipulating
        street view images from the Amsterdam repository.

        Args:
            root_dir:
                An optional custom root directory. Defaults to
                DATA_HOME/sources/amsterdampanorama, where DATA_HOME is read from the
                environment variables. Defaults to None.
        """

        super().__init__(
            root_dir=root_dir,
            url="https://api.data.amsterdam.nl/panorama/panoramas/",
        )

    def get_image_url(self, image_id):
        raise NotImplementedError(
            "get_image_url not implemented for Amsterdam. Use URLs returned by fetch_image_ids directly."
        )

    def fetch_image_ids(self, lat, lon, radius):
        return self._fetch_image_ids(near=f"{lon},{lat}", radius=radius)

    def _fetch_image_ids(self, **params):
        """Fetch panorama IDs and metadata within radius of given point."""
        panoramas = []

        response = self.session.get(self.url, params=params)
        while True:
            response.raise_for_status()
            data = response.json()
            # Process results
            for pano in data["_embedded"]["panoramas"]:
                panoramas.append(
                    {
                        "pano_id": pano["pano_id"],
                        "timestamp": pano["timestamp"],
                        "lon": pano["geometry"]["coordinates"][0],
                        "lat": pano["geometry"]["coordinates"][1],
                        "height": pano["geometry"]["coordinates"][2],
                        "heading": pano["heading"],
                        "roll": pano["roll"],
                        "pitch": pano["pitch"],
                        "thumbnail_url": pano["_links"]["thumbnail"]["href"],
                        "cubic_img_baseurl": pano["cubic_img_baseurl"],
                        "cubic_img_pattern": pano["cubic_img_pattern"],
                        "equirectangular_full": pano["_links"]["equirectangular_full"][
                            "href"
                        ],
                    }
                )

            # Subsequent requests use the "next page" url
            url = data["_links"]["next"]["href"] if data["_links"]["next"] else None
            if url is None:
                break
            response = self.session.get(url)

        return ibis.memtable(panoramas)


# --------------------------------------
# old code I encountered:

# import geopandas as gpd
# import pandas as pd
# import requests
# from shapely import Point

# # https://api.data.amsterdam.nl/panorama/panoramas/?limit_results=100&near=4.9,52.3&radius=200


# def fetch_near(lon, lat, radius, limit=100, **kwargs):
#     return fetch_panoramas(near=f"{lon},{lat}", radius=radius, **kwargs)


# def fetch_panoramas(**params):
#     """Fetch panorama metadata as a GeoDataFrame."""
#     url = "https://api.data.amsterdam.nl/panorama/panoramas/"

#     panoramas = []

#     # First request with initial params
#     next_url = url

#     while next_url:
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         data = response.json()
#         # Process results
#         for pano in data["_embedded"]["panoramas"]:
#             panoramas.append(
#                 {
#                     "pano_id": pano["pano_id"],
#                     "timestamp": pano["timestamp"],
#                     "lon": pano["geometry"]["coordinates"][0],
#                     "lat": pano["geometry"]["coordinates"][1],
#                     "heading": pano["heading"],
#                     "roll": pano["roll"],
#                     "pitch": pano["pitch"],
#                     "thumbnail_url": pano["_links"]["thumbnail"]["href"],
#                     "cubic_img_baseurl": pano["cubic_img_baseurl"],
#                     "cubic_img_pattern": pano["cubic_img_pattern"],
#                     "equirectangular_full": pano["_links"]["equirectangular_full"][
#                         "href"
#                     ],
#                 }
#             )

#         # Subsequent requests use the "next page" url
#         next_url = data["_links"]["next"]["href"] if data["_links"]["next"] else None
#         if not next_url:
#             break

#         response = requests.get(next_url)
#         response.raise_for_status()
#         data = response.json()

#     # Convert to GeoDataFrame
#     df = pd.DataFrame(panoramas)
#     gdf = gpd.GeoDataFrame(
#         df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326"
#     )
#     return gdf


# def download_image(url: str, filename: str) -> None:
#     """Download an image from a given URL and save it to a file."""
#     response = requests.get(url, stream=True)
#     response.raise_for_status()
#     with open(filename, "wb") as file:
#         for chunk in response.iter_content(1024):
#             file.write(chunk)


# if __name__ == "__main__":
#     df = fetch_near(lon=4.9, lat=52.3, radius=200)
#     print(df)
