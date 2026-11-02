# Image sources

Streetscapes can retrieve street view images from several online sources. For each source this is done in two steps with the [CLI](cli.md): fetch the metadata of the images in a bounding box (`west south east north`), then download the images it describes. For example:

```bash
streetscapes fetch-metadata mapillary 4.86 52.32 4.98 52.38 --pano-only --daytime-only
streetscapes download-images mapillary
```

The options of each source can be listed with `streetscapes fetch-metadata <source> --help` and `streetscapes download-images <source> --help`. All sources support the `--pano-only` and `--daytime-only` filters. The metadata and images are stored in the active project.

| Source | CLI command | Token | Downloaded resolution |
|--------|-------------|-------|-----------------------|
| [Mapillary](#mapillary) | `mapillary` | Required | 2048 px wide |
| [KartaView](#kartaview) | `kartaview` | Optional | Full resolution |
| [Panoramax](#panoramax) | `panoramax` | Not needed | Full resolution |

## Mapillary

[Mapillary](https://www.mapillary.com/) is a crowdsourced street view platform, with a large worldwide coverage. Images are retrieved through the [Mapillary API](https://www.mapillary.com/developer/api-documentation). As some areas contain enormous amounts of images, the bounding box is split into tiles (`--tile-size`), with a maximum number of images per tile (`--images-per-tile`). Images are downloaded as the 2048 px wide version Mapillary provides, not the original.

!!! note
    The Mapillary API requires a token. Register at Mapillary, create a token as described in the [API documentation](https://www.mapillary.com/developer/api-documentation), and set it in the streetscapes config:

    ```bash
    streetscapes config set mapillary_token YOUR_TOKEN
    ```

## KartaView

[KartaView](https://kartaview.org/) (formerly OpenStreetCam) is a crowdsourced street view platform linked to OpenStreetMap. Images are retrieved through the [KartaView API](https://doc.kartaview.org/). The API pages through a bounding box of any size, so instead of tiles a single `--image-limit` is used.

!!! note
    A token is not required, but raises the rate limit from 100 to 1000 requests per hour. See the [CLI documentation](cli.md#kartaview-token) for how to obtain a token through OpenStreetMap, and set it with `streetscapes config set kartaview_token YOUR_TOKEN`. KartaView's capture timestamps are known to be unreliable, so `--daytime-only` may keep night-time images or drop daytime ones.

## Panoramax

[Panoramax](https://panoramax.fr/) is a federation of open street view instances, such as the one hosted by [OpenStreetMap France](https://panoramax.openstreetmap.fr). By default, the [federated catalogue](https://docs.panoramax.fr/federated-catalog/) is searched, which covers all participating instances. A single instance can be selected with `--instance`. Images are downloaded from the instance that hosts them, often as 360° panoramas of up to 8000×4000 px. No token is needed.

!!! note
    A single request returns at most 32767 images, so use a small enough `--tile-size` for areas with many images; a fetch that hits this limit reports it.

## Local images

Images that you collected yourself, or that come from a source not supported by streetscapes, can be imported into a project and segmented like any other image. See [importing local images](local-images.md).
