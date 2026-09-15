import io
import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from experimental.overhead_matching.swag.farfield.loci import satellite


_WEB_MERCATOR_LIMIT = 20_037_508.342789244


def _jpeg_bytes(colour: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (256, 256), colour).save(
        output, format="JPEG", quality=95)
    return output.getvalue()


def _png_bytes(colour: tuple[int, int, int], *,
               size: tuple[int, int] = (256, 256)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, colour).save(output, format="PNG")
    return output.getvalue()


def _tiny_grid(*, shape_xy: tuple[int, int] = (2, 2)) -> dict:
    """A self-consistent, four-patch grid with a small source-tile cache."""
    tile_px = 256
    source_px = 640
    stride_px = 320.0
    min_x = 81_200_128.0
    min_y = 99_000_128.0
    n_x, n_y = shape_xy
    last_x = min_x + (n_x - 1) * stride_px
    last_y = min_y + (n_y - 1) * stride_px
    half = source_px / 2.0
    x_min = int((min_x - half) // tile_px)
    y_min = int((min_y - half) // tile_px)
    x_max = int((last_x + half - 1.0) // tile_px)
    y_max = int((last_y + half - 1.0) // tile_px)
    return {
        "schema": "loci_web_mercator_grid/v1",
        "zoom": 20,
        "tile_px": tile_px,
        "patch_px": 640,
        "source_px": source_px,
        "overlap_fraction": 0.5,
        "stride_px": stride_px,
        "min_pixel_xy": [min_x, min_y],
        "max_requested_pixel_xy": [last_x, last_y],
        "last_center_pixel_xy": [last_x, last_y],
        "shape_xy": [n_x, n_y],
        "n_patches": n_x * n_y,
        "center_bbox_wsen": [-71.10, 42.34, -71.09, 42.35],
        "footprint_bbox_wsen": [-71.11, 42.33, -71.08, 42.36],
        "source_tile_range_xyxy": [x_min, y_min, x_max, y_max],
        "n_source_tiles": (x_max - x_min + 1) * (y_max - y_min + 1),
        "metres_per_pixel_at_mid_lat": 0.11,
        "patch_ground_m_at_mid_lat": 70.4,
        "stride_ground_m_at_mid_lat": 35.2,
    }


def _grid_with_source_shape(width: int, height: int) -> dict:
    grid = _tiny_grid(shape_xy=(1, 1))
    x_min, y_min, _, _ = grid["source_tile_range_xyxy"]
    grid["source_tile_range_xyxy"] = [
        x_min, y_min, x_min + width - 1, y_min + height - 1]
    grid["n_source_tiles"] = width * height
    return grid


def _service_metadata() -> dict:
    return {
        "name": "Massachusetts_Aerial_Imagery_2025",
        "capabilities": "Map,Tilemap,Query",
        "serviceItemId": "test-service-item",
        "minLOD": 0,
        "maxLOD": 20,
        "fullExtent": {
            "xmin": -_WEB_MERCATOR_LIMIT,
            "ymin": -_WEB_MERCATOR_LIMIT,
            "xmax": _WEB_MERCATOR_LIMIT,
            "ymax": _WEB_MERCATOR_LIMIT,
            "spatialReference": {"wkid": 102100, "latestWkid": 3857},
        },
        "tileInfo": {
            "rows": 256,
            "cols": 256,
            "format": "JPG",
            "compressionQuality": 90,
            "spatialReference": {"wkid": 102100, "latestWkid": 3857},
            "lods": [{
                "level": 20,
                "resolution": 156543.03392804097 / (2 ** 20),
                "scale": 591657527.591555 / (2 ** 20),
            }],
        },
    }


def _image_server_metadata() -> dict:
    return {
        "name": "USGSNAIPImagery",
        "capabilities": "Image,Metadata,Catalog,Mensuration",
        "serviceItemId": None,
        "fullExtent": {
            "xmin": -_WEB_MERCATOR_LIMIT,
            "ymin": -_WEB_MERCATOR_LIMIT,
            "xmax": _WEB_MERCATOR_LIMIT,
            "ymax": _WEB_MERCATOR_LIMIT,
            "spatialReference": {"wkid": 102100, "latestWkid": 3857},
        },
        "maxImageWidth": 4000,
        "maxImageHeight": 4000,
        "pixelSizeX": 0.3,
        "pixelSizeY": 0.3,
        "meanPixelSize": 0.3,
        "bandCount": 4,
        "pixelType": "U8",
        "description": "test NAIP imagery",
        "copyrightText": "USGS, USDA, The National Map",
    }


def _source_index_covering(bbox_wsen: list[float]) -> dict:
    west, south, east, north = bbox_wsen
    return {
        "features": [{
            "attributes": {
                "OBJECTID": 1,
                "TILE_NAME": "TEST_2025",
                "ZONE": "TEST",
                "URL": "https://example.invalid/test.tif",
            },
            "geometry": {"rings": [[
                [west, south], [east, south], [east, north],
                [west, north], [west, south],
            ]]},
        }],
        "exceededTransferLimit": False,
    }


def _image_catalog_covering(
        bbox_wsen: list[float], raster_ids: tuple[int, ...]) -> dict:
    west, south, east, north = bbox_wsen
    return {
        "objectIdFieldName": "OBJECTID",
        "geometryType": "esriGeometryPolygon",
        "spatialReference": {"wkid": 4326, "latestWkid": 4326},
        "features": [{
            "attributes": {
                "OBJECTID": raster_id,
                "Name": f"m_test_030_20230901_{raster_id}",
                "State": "NH",
                "Year": 2023,
                "raster_name": f"m_test_030_20230901_{raster_id}",
                "download_url": (
                    f"https://example.invalid/{raster_id}.tif"),
                "acquisition_date": 1_693_526_400_000,
                "agency": "USDA",
                "vendor": "USDA-FSA-APFO",
                "resolution_value": 0.3,
                "resolution_units": "METER",
                "band_count": 4,
                "sensor_type": "CNIR",
                "Category": 1,
            },
            "geometry": {"rings": [[
                [west, south], [east, south], [east, north],
                [west, north], [west, south],
            ]]},
        } for raster_id in raster_ids],
        "exceededTransferLimit": False,
    }


class FakeTileClient:
    def __init__(self, *, missing_tiles=(), tilemap_missing=(), delay=0.0):
        self.missing_tiles = set(missing_tiles)
        self.tilemap_missing = set(tilemap_missing)
        self.delay = delay
        self.fetch_calls: list[tuple[int, int, int]] = []
        self.tilemap_calls: list[tuple[int, int, int, int, int]] = []
        self.source_index_calls: list[tuple[str, tuple[float, ...]]] = []
        self._lock = threading.Lock()
        self.active_fetches = 0
        self.max_active_fetches = 0

    def get_service_metadata(self) -> dict:
        return _service_metadata()

    def fetch_tile(self, zoom: int, tile_x: int, tile_y: int) -> bytes:
        key = (tile_x, tile_y)
        with self._lock:
            self.fetch_calls.append((zoom, tile_x, tile_y))
            self.active_fetches += 1
            self.max_active_fetches = max(
                self.max_active_fetches, self.active_fetches)
        try:
            if self.delay:
                time.sleep(self.delay)
            if key in self.missing_tiles:
                raise satellite.SatelliteError(
                    f"missing fake source tile {zoom}/{tile_y}/{tile_x}")
            return _jpeg_bytes((tile_x % 251, tile_y % 251,
                                (tile_x + tile_y) % 251))
        finally:
            with self._lock:
                self.active_fetches -= 1

    def get_tilemap(self, zoom: int, tile_x: int, tile_y: int,
                    width: int, height: int) -> dict:
        self.tilemap_calls.append((zoom, tile_x, tile_y, width, height))
        data = [
            0 if (x, y) in self.tilemap_missing else 1
            for y in range(tile_y, tile_y + height)
            for x in range(tile_x, tile_x + width)
        ]
        return {
            "adjusted": False,
            "location": {
                "left": tile_x,
                "top": tile_y,
                "width": width,
                "height": height,
            },
            "data": data,
        }

    def query_source_index(self, url: str,
                           bbox_wsen: list[float]) -> dict:
        self.source_index_calls.append((url, tuple(bbox_wsen)))
        return _source_index_covering(bbox_wsen)


class FakeImageServerClient:
    def __init__(self, catalog_response: dict):
        self.catalog_response = catalog_response
        self.catalog_calls: list[tuple[float, ...]] = []
        self.fetch_calls: list[tuple[int, int, int]] = []

    def get_service_metadata(self) -> dict:
        return _image_server_metadata()

    def query_catalog(self, bbox_wsen: list[float]) -> dict:
        self.catalog_calls.append(tuple(bbox_wsen))
        return self.catalog_response

    def fetch_tile(self, zoom: int, tile_x: int, tile_y: int) -> bytes:
        self.fetch_calls.append((zoom, tile_x, tile_y))
        return _png_bytes((tile_x % 251, tile_y % 251,
                           (tile_x + tile_y) % 251))


class FakeChunkImageServerClient:
    def __init__(self, *, delay: float = 0.0):
        self.delay = delay
        self.chunk_calls: list[tuple[int, int, int, int, int]] = []
        self._lock = threading.Lock()
        self.active_fetches = 0
        self.max_active_fetches = 0

    @staticmethod
    def colour(tile_x: int, tile_y: int) -> tuple[int, int, int]:
        return (
            tile_x * 17 % 250 + 1,
            tile_y * 29 % 250 + 1,
            (tile_x + tile_y) * 37 % 250 + 1,
        )

    def fetch_tile_chunk(self, zoom: int, tile_x: int, tile_y: int,
                         width: int, height: int) \
            -> satellite.ImageServerTileChunk:
        with self._lock:
            self.chunk_calls.append(
                (zoom, tile_x, tile_y, width, height))
            self.active_fetches += 1
            self.max_active_fetches = max(
                self.max_active_fetches, self.active_fetches)
        try:
            if self.delay:
                time.sleep(self.delay)
            tile_px = 256
            composite = Image.new(
                "RGB", (width * tile_px, height * tile_px))
            tiles = []
            for offset_y in range(height):
                for offset_x in range(width):
                    child_x = tile_x + offset_x
                    child_y = tile_y + offset_y
                    child = Image.new(
                        "RGB", (tile_px, tile_px),
                        self.colour(child_x, child_y))
                    composite.paste(
                        child, (offset_x * tile_px, offset_y * tile_px))
                    value, info = satellite._canonical_rgb_png_bytes(child)
                    tiles.append(satellite.ImageServerSourceTile(
                        child_x, child_y, value, info))
            output = io.BytesIO()
            composite.save(output, format="PNG")
            response_value = output.getvalue()
            _, response_info = satellite._validate_image_bytes(
                response_value, composite.size, "fake composite")
            return satellite.ImageServerTileChunk(
                zoom, tile_x, tile_y, width, height,
                response_info, tuple(tiles))
        finally:
            with self._lock:
                self.active_fetches -= 1


class SatelliteTest(unittest.TestCase):
    def test_wms_chunk_reprojects_to_exact_grid_and_resumes(self):
        layer = "pohang_2022_1225cm"
        service_url = "https://example.invalid/wms"
        capabilities = f"""<?xml version="1.0"?>
<WMT_MS_Capabilities version="1.1.1">
  <Service><Name>OGC:WMS</Name><Title>Pohang Airmap</Title>
    <Fees>none</Fees><AccessConstraints>none</AccessConstraints></Service>
  <Capability><Request><GetMap><Format>image/jpeg</Format></GetMap></Request>
    <Layer><Title>Pohang Airmap</Title><SRS>EPSG:5186</SRS>
      <Layer><Name>{layer}</Name><Title>{layer}</Title>
        <LatLonBoundingBox minx="128.974" miny="35.823"
                           maxx="129.604" maxy="36.351"/>
        <BoundingBox SRS="EPSG:5186" minx="377235" miny="360461"
                     maxx="434901" maxy="419953"/>
      </Layer>
    </Layer>
  </Capability>
</WMT_MS_Capabilities>""".encode()
        client = satellite.OgcWmsClient(
            service_url, layer=layer, srs="EPSG:5186", chunk_tiles=7)

        def response(_url, *, params):
            if params["REQUEST"] == "GetCapabilities":
                return SimpleNamespace(
                    headers={"Content-Type":
                             "application/vnd.ogc.wms_xml;charset=UTF-8"},
                    content=capabilities)
            size = (int(params["WIDTH"]), int(params["HEIGHT"]))
            encoded = io.BytesIO()
            Image.new("RGB", size, (17, 83, 141)).save(
                encoded, format="JPEG", quality=95)
            return SimpleNamespace(
                headers={"Content-Type": "image/jpeg;charset=UTF-8"},
                content=encoded.getvalue())

        client._get = mock.Mock(side_effect=response)
        metadata = client.get_service_metadata()
        center_x, center_y = satellite.region.lat_lon_to_pixel(
            36.03, 129.38, 20)
        grid = _tiny_grid(shape_xy=(1, 1))
        grid["min_pixel_xy"] = [center_x, center_y]
        grid["max_requested_pixel_xy"] = [center_x, center_y]
        grid["last_center_pixel_xy"] = [center_x, center_y]
        origin_x = satellite.region.nearest_pixel_origin(center_x, 640)
        origin_y = satellite.region.nearest_pixel_origin(center_y, 640)
        source_range = [
            origin_x // 256, origin_y // 256,
            (origin_x + 639) // 256, (origin_y + 639) // 256,
        ]
        grid["source_tile_range_xyxy"] = source_range
        grid["n_source_tiles"] = (
            (source_range[2] - source_range[0] + 1)
            * (source_range[3] - source_range[1] + 1))
        grid["footprint_bbox_wsen"] = \
            satellite._rendered_footprint_bbox_wsen(grid)
        audit = satellite.audit_coverage(
            client, {"grid": grid}, metadata,
            service_url=service_url, source_index_url=None,
            provider_mode=satellite.WMS_PROVIDER,
            wms_layer=layer, wms_srs="EPSG:5186", wms_chunk_tiles=7)
        self.assertEqual(
            audit["provider"]["capabilities"]["response_sha256"],
            satellite._sha256_bytes(capabilities))
        self.assertEqual(
            audit["provider"]["no_data"]["policy"],
            "preserve_source_pixels_no_color_key_v1")

        with TemporaryDirectory() as temporary:
            build_dir = Path(temporary)
            first = satellite.ensure_source_tiles(
                build_dir, grid, client, workers=2,
                source_chunking_contract=satellite._wms_chunk_contract(7))
            self.assertEqual(first["downloaded"], grid["n_source_tiles"])
            get_map_calls = [
                call for call in client._get.call_args_list
                if call.kwargs["params"]["REQUEST"] == "GetMap"
            ]
            self.assertEqual(len(get_map_calls), 1)
            parameters = get_map_calls[0].kwargs["params"]
            self.assertEqual(parameters["SRS"], "EPSG:5186")
            self.assertEqual(parameters["FORMAT"], "image/jpeg")
            self.assertLessEqual(int(parameters["WIDTH"]), 1792)
            self.assertLessEqual(int(parameters["HEIGHT"]), 1792)
            receipt_path, = (build_dir / "source_tile_chunks").rglob(
                "*.json")
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["schema"], satellite.WMS_CHUNK_SCHEMA)
            first_tile = satellite._tile_cache_path(
                build_dir, 20, source_range[0], source_range[1])
            with Image.open(first_tile) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (256, 256))
                self.assertLess(
                    max(abs(found - expected) for found, expected in zip(
                        image.getpixel((128, 128)), (17, 83, 141))), 3)

            resumed = satellite.ensure_source_tiles(
                build_dir, grid, client, workers=2,
                source_chunking_contract=satellite._wms_chunk_contract(7))
            self.assertEqual(resumed["resumed"], grid["n_source_tiles"])
            self.assertEqual(
                len([call for call in client._get.call_args_list
                     if call.kwargs["params"]["REQUEST"] == "GetMap"]), 1)

    def test_esri_wayback_release_pins_tile_urls_and_provider_identity(self):
        client = satellite.ArcGisTileClient(
            satellite.ESRI_WORLD_IMAGERY_SERVICE_URL,
            esri_wayback_release="32246")
        client.get_json = mock.Mock(return_value={})

        client.get_service_metadata()
        client.get_json.assert_called_once_with(
            satellite.ESRI_WORLD_IMAGERY_SERVICE_URL, params={"f": "json"})
        client.get_json.reset_mock()
        client.get_tilemap(20, 540_219, 344_179, 69, 13)
        client.get_json.assert_called_once_with(
            satellite.ESRI_WAYBACK_TILE_SERVICE_URL
            + "/tilemap/32246/20/344179/540219/69/13",
            params={"f": "json"})

        client._get = mock.Mock(return_value=SimpleNamespace(content=b"tile"))
        self.assertEqual(client.fetch_tile(20, 540_219, 344_179), b"tile")
        client._get.assert_called_once_with(
            satellite.ESRI_WAYBACK_TILE_SERVICE_URL
            + "/tile/32246/20/344179/540219",
            params={"blankTile": "false"}, missing_is_error=True)

        request = satellite._provider_request_contract(
            satellite.CACHED_MAP_PROVIDER,
            satellite.ESRI_WORLD_IMAGERY_SERVICE_URL,
            source_index_url=None, catalog_where=None, lock_raster_ids=(),
            esri_wayback_release="32246")
        metadata = _service_metadata()
        for key in ("name", "minLOD", "maxLOD"):
            metadata.pop(key)
        provider = satellite._cached_map_provider_contract(
            satellite.ESRI_WORLD_IMAGERY_SERVICE_URL,
            metadata, _tiny_grid(),
            esri_wayback_release=request["esri_wayback_release"])
        self.assertEqual(request["esri_wayback_release"], "32246")
        self.assertEqual(provider["esri_wayback_release"], "32246")
        self.assertEqual(provider["service_name"], "World_Imagery")
        self.assertEqual(provider["min_lod"], 20)
        self.assertEqual(provider["max_lod"], 20)
        self.assertEqual(
            provider["tile_service_url"],
            satellite.ESRI_WAYBACK_TILE_SERVICE_URL)

    def test_fractional_patch_origin_fills_every_pixel(self):
        grid = _tiny_grid(shape_xy=(1, 1))
        colour = (37, 89, 143)
        tile_cache = mock.Mock()
        tile_cache.get.return_value = Image.new(
            "RGB", (grid["tile_px"], grid["tile_px"]), colour)

        patch = satellite.assemble_patch(
            grid["min_pixel_xy"][0] + 0.82231614,
            grid["min_pixel_xy"][1] + 0.22762978,
            grid,
            tile_cache,
        )

        self.assertEqual(patch.getextrema(), tuple((v, v) for v in colour))
        self.assertEqual(patch.getpixel((639, 639)), colour)

    def test_validate_image_file_rejects_corrupt_and_wrong_sized_images(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.jpg"
            valid.write_bytes(_jpeg_bytes((10, 20, 30)))
            self.assertIsNotNone(
                satellite.validate_image_file(valid, (256, 256)))

            valid.write_bytes(b"not an image")
            self.assertIsNone(
                satellite.validate_image_file(valid, (256, 256)))

            wrong_size = root / "wrong-size.jpg"
            Image.new("RGB", (32, 32)).save(wrong_size, "JPEG")
            self.assertIsNone(
                satellite.validate_image_file(wrong_size, (256, 256)))

    def test_image_server_export_request_is_pinned_north_up_png(self):
        raster_ids = (30, 10, 20)
        client = satellite.ArcGisImageServerClient(
            "https://example.invalid/ImageServer",
            catalog_where="State='NH' AND Year=2023 AND Category=1",
            lock_raster_ids=raster_ids)
        response = SimpleNamespace(
            headers={"Content-Type": "image/png; charset=binary"},
            content=_png_bytes((10, 20, 30)))
        client._get = mock.Mock(return_value=response)

        value = client.fetch_tile(19, 158_288, 190_104)

        self.assertEqual(value, response.content)
        url, = client._get.call_args.args
        parameters = client._get.call_args.kwargs["params"]
        self.assertEqual(
            url, "https://example.invalid/ImageServer/exportImage")
        self.assertEqual(parameters["bboxSR"], "3857")
        self.assertEqual(parameters["imageSR"], "3857")
        self.assertEqual(parameters["size"], "256,256")
        self.assertEqual(parameters["format"], "png")
        self.assertEqual(parameters["f"], "image")
        self.assertEqual(
            json.loads(parameters["renderingRule"]),
            {"rasterFunction": "NaturalColor"})
        self.assertEqual(json.loads(parameters["mosaicRule"]), {
            "mosaicMethod": "esriMosaicLockRaster",
            "lockRasterIds": [10, 20, 30],
            "mosaicOperation": "MT_FIRST",
        })
        bbox = tuple(float(item) for item in parameters["bbox"].split(","))
        expected = (
            -7_938_444.009585265,
            5_506_447.080635812,
            -7_938_367.572556980,
            5_506_523.517664097,
        )
        for found, wanted in zip(bbox, expected):
            self.assertAlmostEqual(found, wanted, places=6)
        south_bbox = satellite._web_mercator_tile_bbox(
            19, 158_288, 190_105)
        self.assertAlmostEqual(bbox[1], south_bbox[3], places=8)
        self.assertLess(south_bbox[1], bbox[1])

        client._get.return_value = SimpleNamespace(
            headers={"Content-Type": "application/json"},
            content=json.dumps({"error": {"code": 400}}).encode())
        with self.assertRaisesRegex(
                satellite.SatelliteError, "JSON instead of imagery"):
            client.fetch_tile(19, 158_288, 190_104)

        client._get.return_value = SimpleNamespace(
            headers={"Content-Type": "image/png"},
            content=_png_bytes((10, 20, 30), size=(32, 32)))
        with self.assertRaisesRegex(satellite.SatelliteError, "expected"):
            client.fetch_tile(19, 158_288, 190_104)

        transparent = io.BytesIO()
        Image.new("RGBA", (256, 256), (10, 20, 30, 0)).save(
            transparent, format="PNG")
        client._get.return_value = SimpleNamespace(
            headers={"Content-Type": "image/png"},
            content=transparent.getvalue())
        with self.assertRaisesRegex(satellite.SatelliteError, "expected RGB"):
            client.fetch_tile(19, 158_288, 190_104)

        client._get.return_value = SimpleNamespace(
            headers={"Content-Type": "image/png"},
            content=_png_bytes((0, 0, 0)))
        with self.assertRaisesRegex(
                satellite.MissingTileError, "all-black no-data"):
            client.fetch_tile(19, 158_288, 190_104)

    def test_image_server_chunk_export_bbox_and_pixel_split_are_exact(self):
        client = satellite.ArcGisImageServerClient(
            "https://example.invalid/ImageServer",
            catalog_where="State='NH'",
            lock_raster_ids=(30, 10))
        tile_x = 158_288
        tile_y = 190_104
        tile_px = 256
        composite = Image.new("RGB", (2 * tile_px, 2 * tile_px))
        colours = {
            (0, 0): (11, 21, 31),
            (1, 0): (41, 51, 61),
            (0, 1): (71, 81, 91),
            (1, 1): (101, 111, 121),
        }
        for (offset_x, offset_y), colour in colours.items():
            composite.paste(
                colour,
                (offset_x * tile_px, offset_y * tile_px,
                 (offset_x + 1) * tile_px,
                 (offset_y + 1) * tile_px))
        encoded = io.BytesIO()
        composite.save(encoded, format="PNG")
        client._get = mock.Mock(return_value=SimpleNamespace(
            headers={"Content-Type": "image/png"},
            content=encoded.getvalue()))

        chunk = client.fetch_tile_chunk(
            19, tile_x, tile_y, width=2, height=2)

        parameters = client._get.call_args.kwargs["params"]
        self.assertEqual(parameters["size"], "512,512")
        bbox = tuple(float(value)
                     for value in parameters["bbox"].split(","))
        northwest = satellite._web_mercator_tile_bbox(
            19, tile_x, tile_y)
        southeast = satellite._web_mercator_tile_bbox(
            19, tile_x + 1, tile_y + 1)
        expected_bbox = (
            northwest[0], southeast[1], southeast[2], northwest[3])
        for found, expected in zip(bbox, expected_bbox):
            self.assertAlmostEqual(found, expected, places=8)
        self.assertEqual(
            [(tile.tile_x, tile.tile_y) for tile in chunk.tiles],
            [(tile_x, tile_y), (tile_x + 1, tile_y),
             (tile_x, tile_y + 1), (tile_x + 1, tile_y + 1)])
        for tile in chunk.tiles:
            with Image.open(io.BytesIO(tile.value)) as child:
                child.load()
                offset = (tile.tile_x - tile_x, tile.tile_y - tile_y)
                self.assertEqual(child.mode, "RGB")
                self.assertEqual(child.size, (tile_px, tile_px))
                self.assertEqual(child.getpixel((0, 0)), colours[offset])
                self.assertEqual(child.getpixel((255, 255)), colours[offset])
                canonical, info = satellite._canonical_rgb_png_bytes(child)
            self.assertEqual(tile.value, canonical)
            self.assertEqual(tile.info, info)

        maximum = satellite._image_server_export_parameters(
            19, tile_x, tile_y, (10,), width=15, height=15)
        self.assertEqual(maximum["size"], "3840,3840")
        with self.assertRaisesRegex(satellite.SatelliteError, "1..15"):
            satellite._image_server_export_parameters(
                19, tile_x, tile_y, (10,), width=16, height=1)

    def test_image_server_chunk_rejects_an_all_black_child(self):
        client = satellite.ArcGisImageServerClient(
            "https://example.invalid/ImageServer",
            catalog_where="State='NH'", lock_raster_ids=(10,))
        composite = Image.new("RGB", (512, 256))
        composite.paste((10, 20, 30), (0, 0, 256, 256))
        encoded = io.BytesIO()
        composite.save(encoded, format="PNG")
        client._get = mock.Mock(return_value=SimpleNamespace(
            headers={"Content-Type": "image/png"},
            content=encoded.getvalue()))

        with self.assertRaisesRegex(
                satellite.MissingTileError, "all-black no-data tile"):
            client.fetch_tile_chunk(19, 158_288, 190_104, 2, 1)

    def test_image_server_chunk_partition_is_fixed_and_edge_aware(self):
        grid = _grid_with_source_shape(31, 17)
        x_min, y_min, _, _ = grid["source_tile_range_xyxy"]

        chunks = list(satellite._iter_image_server_source_chunks(grid, 15))

        self.assertEqual(chunks, [
            (x_min, y_min, 15, 15),
            (x_min + 15, y_min, 15, 15),
            (x_min + 30, y_min, 1, 15),
            (x_min, y_min + 15, 15, 2),
            (x_min + 15, y_min + 15, 15, 2),
            (x_min + 30, y_min + 15, 1, 2),
        ])

    def test_rendered_footprint_uses_quantized_crop_edges(self):
        grid = _tiny_grid(shape_xy=(2, 2))
        grid["min_pixel_xy"] = [
            grid["min_pixel_xy"][0] + 0.49,
            grid["min_pixel_xy"][1] - 0.49,
        ]

        west, south, east, north = \
            satellite._rendered_footprint_bbox_wsen(grid)
        west_pixel, north_pixel = satellite.region.lat_lon_to_pixel(
            north, west, grid["zoom"])
        east_pixel, south_pixel = satellite.region.lat_lon_to_pixel(
            south, east, grid["zoom"])
        first_x = satellite.region.nearest_pixel_origin(
            grid["min_pixel_xy"][0], grid["source_px"])
        first_y = satellite.region.nearest_pixel_origin(
            grid["min_pixel_xy"][1], grid["source_px"])
        last_x = satellite.region.nearest_pixel_origin(
            grid["min_pixel_xy"][0] + grid["stride_px"],
            grid["source_px"])
        last_y = satellite.region.nearest_pixel_origin(
            grid["min_pixel_xy"][1] + grid["stride_px"],
            grid["source_px"])
        self.assertAlmostEqual(west_pixel, first_x, places=6)
        self.assertAlmostEqual(north_pixel, first_y, places=6)
        self.assertAlmostEqual(
            east_pixel, last_x + grid["source_px"], places=6)
        self.assertAlmostEqual(
            south_pixel, last_y + grid["source_px"], places=6)

    def test_image_server_catalog_requires_coverage_and_exact_locks(self):
        grid = _tiny_grid()
        bbox = satellite._rendered_footprint_bbox_wsen(grid)
        raster_ids = (10, 20)
        clause = "State='NH' AND Year=2023 AND Category=1"
        complete = FakeImageServerClient(
            _image_catalog_covering(bbox, raster_ids))

        audit = satellite.audit_coverage(
            complete, {"grid": grid}, _image_server_metadata(),
            service_url="https://example.invalid/ImageServer",
            source_index_url=None,
            provider_mode=satellite.IMAGE_SERVER_PROVIDER,
            catalog_where=clause, lock_raster_ids=reversed(raster_ids))

        self.assertEqual(audit["tilemap"]["status"], "not_applicable")
        self.assertTrue(audit["catalog"]["covers_footprint"])
        self.assertEqual(audit["catalog"]["lock_raster_ids"], [10, 20])
        self.assertEqual(
            [item["object_id"] for item in audit["catalog"]["source_rasters"]],
            [10, 20])
        self.assertIn("rings", audit["catalog"]["source_rasters"][0]
                      ["geometry"])
        self.assertEqual(
            audit["catalog"]["query_parameters"]["where"], clause)
        self.assertEqual(
            audit["provider"]["export"]["mosaic_rule"]
            ["lockRasterIds"], [10, 20])
        self.assertEqual(
            audit["rendered_footprint_bbox_wsen"],
            satellite._rendered_footprint_bbox_wsen(grid))

        with self.assertRaisesRegex(
                satellite.SatelliteError, "always strict"):
            satellite.audit_coverage(
                complete, {"grid": grid}, _image_server_metadata(),
                service_url="https://example.invalid/ImageServer",
                source_index_url=None,
                require_source_index_coverage=False,
                provider_mode=satellite.IMAGE_SERVER_PROVIDER,
                catalog_where=clause, lock_raster_ids=raster_ids)

        west, south, east, north = bbox
        gap = FakeImageServerClient(_image_catalog_covering(
            [west, south, (west + east) / 2.0, north], raster_ids))
        with self.assertRaisesRegex(
                satellite.SatelliteError, "does not cover"):
            satellite.audit_coverage(
                gap, {"grid": grid}, _image_server_metadata(),
                service_url="https://example.invalid/ImageServer",
                source_index_url=None,
                provider_mode=satellite.IMAGE_SERVER_PROVIDER,
                catalog_where=clause, lock_raster_ids=raster_ids)

        mismatch = FakeImageServerClient(
            _image_catalog_covering(bbox, (10,)))
        with self.assertRaisesRegex(
                satellite.SatelliteError, "does not match lock"):
            satellite.audit_coverage(
                mismatch, {"grid": grid}, _image_server_metadata(),
                service_url="https://example.invalid/ImageServer",
                source_index_url=None,
                provider_mode=satellite.IMAGE_SERVER_PROVIDER,
                catalog_where=clause, lock_raster_ids=raster_ids)

    def test_image_server_chunk_requires_full_3840px_export_capacity(self):
        grid = _tiny_grid()
        bbox = satellite._rendered_footprint_bbox_wsen(grid)
        clause = "State='NH' AND Year=2023 AND Category=1"
        raster_ids = (10,)
        client = FakeImageServerClient(
            _image_catalog_covering(bbox, raster_ids))

        accepted = satellite.audit_coverage(
            client, {"grid": grid}, _image_server_metadata(),
            service_url="https://example.invalid/ImageServer",
            source_index_url=None,
            provider_mode=satellite.IMAGE_SERVER_PROVIDER,
            catalog_where=clause, lock_raster_ids=raster_ids,
            image_server_chunk_tiles=15)
        self.assertEqual(
            accepted["provider"]["max_image_width"], 4000)
        self.assertEqual(
            accepted["provider"]["max_image_height"], 4000)

        for dimension in ("maxImageWidth", "maxImageHeight"):
            with self.subTest(dimension=dimension):
                metadata = _image_server_metadata()
                metadata[dimension] = 3839
                with self.assertRaisesRegex(
                        satellite.SatelliteError,
                        "smaller than the requested 15x15"):
                    satellite.audit_coverage(
                        client, {"grid": grid}, metadata,
                        service_url="https://example.invalid/ImageServer",
                        source_index_url=None,
                        provider_mode=satellite.IMAGE_SERVER_PROVIDER,
                        catalog_where=clause, lock_raster_ids=raster_ids,
                        image_server_chunk_tiles=15)

    def test_image_server_identity_distinguishes_pinned_rasters(self):
        grid = _tiny_grid()
        bbox = satellite._rendered_footprint_bbox_wsen(grid)
        clause = "State='NH' AND Year=2023 AND Category=1"

        def audited(raster_id: int) -> dict:
            client = FakeImageServerClient(
                _image_catalog_covering(bbox, (raster_id,)))
            return satellite.audit_coverage(
                client, {"grid": grid}, _image_server_metadata(),
                service_url="https://example.invalid/ImageServer",
                source_index_url=None,
                provider_mode=satellite.IMAGE_SERVER_PROVIDER,
                catalog_where=clause, lock_raster_ids=(raster_id,))

        first = audited(10)
        second = audited(20)
        chunk_client = FakeImageServerClient(
            _image_catalog_covering(bbox, (10,)))
        chunked = satellite.audit_coverage(
            chunk_client, {"grid": grid}, _image_server_metadata(),
            service_url="https://example.invalid/ImageServer",
            source_index_url=None,
            provider_mode=satellite.IMAGE_SERVER_PROVIDER,
            catalog_where=clause, lock_raster_ids=(10,),
            image_server_chunk_tiles=15)
        self.assertNotIn("export_chunking", first["provider_request"])
        self.assertNotIn("chunking", first["provider"]["export"])
        self.assertEqual(
            chunked["provider_request"]["export_chunking"]
            ["shape_tiles_xy"], [15, 15])
        self.assertEqual(
            chunked["provider"]["export"]["chunking"]
            ["child_encoding"],
            satellite.IMAGE_SERVER_SOURCE_TILE_ENCODING)
        self.assertNotEqual(first["provider_request"],
                            chunked["provider_request"])
        self.assertNotEqual(
            satellite.artifact.sha256_json(first["provider"]),
            satellite.artifact.sha256_json(second["provider"]))
        self.assertNotEqual(
            satellite.artifact.sha256_json(first),
            satellite.artifact.sha256_json(second))

        with TemporaryDirectory() as temporary:
            build_dir = Path(temporary) / "build"
            state = {
                "schema": satellite.BUILD_SCHEMA,
                "dataset": "tiny",
                "version": "naip_v1",
                "region_manifest_digest": "a" * 64,
                "region_content_digest": "b" * 64,
                "provider": first["provider"],
                "provider_request": first["provider_request"],
            }
            satellite._prepare_build_directory(build_dir, state)
            with self.assertRaisesRegex(
                    satellite.SatelliteError, "different recipe"):
                satellite._prepare_build_directory(build_dir, {
                    **state,
                    "provider": second["provider"],
                    "provider_request": second["provider_request"],
                })
            with self.assertRaisesRegex(
                    satellite.SatelliteError, "different recipe"):
                satellite._prepare_build_directory(build_dir, {
                    **state,
                    "provider": chunked["provider"],
                    "provider_request": chunked["provider_request"],
                })

            destination = Path(temporary) / "existing"
            destination.mkdir()
            reference = SimpleNamespace(dataset="tiny")
            region_reference = SimpleNamespace(
                dataset="tiny", manifest_digest="a" * 64)
            manifest = SimpleNamespace(config={
                "region_manifest_digest": "a" * 64,
                "provider": first["provider"],
                "provider_request": first["provider_request"],
                "source_index_url": None,
                "jpeg_quality": satellite.DEFAULT_JPEG_QUALITY,
                "assembly_version": satellite.ASSEMBLY_VERSION,
            })
            with mock.patch.object(
                    satellite.artifact, "open_artifact",
                    return_value=reference), mock.patch.object(
                        satellite.artifact, "load_manifest",
                        return_value=manifest):
                self.assertIs(
                    satellite._existing_artifact(
                        destination, region_reference,
                        service_url="https://example.invalid/ImageServer",
                        source_index_url=None,
                        require_source_index_coverage=True,
                        jpeg_quality=satellite.DEFAULT_JPEG_QUALITY,
                        provider_mode=satellite.IMAGE_SERVER_PROVIDER,
                        catalog_where=clause, lock_raster_ids=(10,)),
                    reference)
                with self.assertRaisesRegex(
                        satellite.SatelliteError, "differs from request"):
                    satellite._existing_artifact(
                        destination, region_reference,
                        service_url="https://example.invalid/ImageServer",
                        source_index_url=None,
                        require_source_index_coverage=True,
                        jpeg_quality=satellite.DEFAULT_JPEG_QUALITY,
                        provider_mode=satellite.IMAGE_SERVER_PROVIDER,
                        catalog_where=clause, lock_raster_ids=(10,),
                        image_server_chunk_tiles=15)
                with self.assertRaisesRegex(
                        satellite.SatelliteError, "differs from request"):
                    satellite._existing_artifact(
                        destination, region_reference,
                        service_url="https://example.invalid/ImageServer",
                        source_index_url=None,
                        require_source_index_coverage=True,
                        jpeg_quality=satellite.DEFAULT_JPEG_QUALITY,
                        provider_mode=satellite.IMAGE_SERVER_PROVIDER,
                        catalog_where=clause, lock_raster_ids=(20,))

                changed_request = SimpleNamespace(config={
                    **manifest.config,
                    "provider_request": {
                        **first["provider_request"],
                        "export_format": "jpg",
                    },
                })
                with mock.patch.object(
                        satellite.artifact, "load_manifest",
                        return_value=changed_request), self.assertRaisesRegex(
                            satellite.SatelliteError,
                            "differs from request"):
                    satellite._existing_artifact(
                        destination, region_reference,
                        service_url="https://example.invalid/ImageServer",
                        source_index_url=None,
                        require_source_index_coverage=True,
                        jpeg_quality=satellite.DEFAULT_JPEG_QUALITY,
                        provider_mode=satellite.IMAGE_SERVER_PROVIDER,
                        catalog_where=clause, lock_raster_ids=(10,))

    def test_audit_coverage_requires_every_source_tile(self):
        grid = _tiny_grid()
        plan = {"grid": grid}
        x_min, y_min, _, _ = grid["source_tile_range_xyxy"]

        complete = FakeTileClient()
        audit = satellite.audit_coverage(
            complete, plan, _service_metadata(),
            service_url="https://example.invalid/MapServer",
            source_index_url="https://example.invalid/source-index")
        self.assertEqual(audit["tilemap"]["n_missing"], 0)
        self.assertNotIn("provider_request", audit)
        self.assertEqual(
            audit["source_index"]["uncovered_coordinate_area"], 0.0)

        incomplete = FakeTileClient(tilemap_missing={(x_min, y_min)})
        with self.assertRaisesRegex(satellite.SatelliteError, "missing"):
            satellite.audit_coverage(
                incomplete, plan, _service_metadata(),
                service_url="https://example.invalid/MapServer")

    def test_source_index_can_be_informational_for_water_gaps(self):
        grid = _tiny_grid()
        plan = {"grid": grid}
        west, south, east, north = grid["footprint_bbox_wsen"]
        client = FakeTileClient()
        client.query_source_index = mock.Mock(return_value=
            _source_index_covering([
                west, south, (west + east) / 2.0, north]))

        with self.assertRaisesRegex(
                satellite.SatelliteError, "does not cover"):
            satellite.audit_coverage(
                client, plan, _service_metadata(),
                service_url="https://example.invalid/MapServer",
                source_index_url="https://example.invalid/source-index")

        audit = satellite.audit_coverage(
            client, plan, _service_metadata(),
            service_url="https://example.invalid/MapServer",
            source_index_url="https://example.invalid/source-index",
            require_source_index_coverage=False)
        source_index = audit["source_index"]
        self.assertEqual(source_index["status"], "informational_partial")
        self.assertFalse(source_index["coverage_required"])
        self.assertFalse(source_index["covers_footprint"])
        self.assertGreater(source_index["uncovered_coordinate_area"], 0.0)

    def test_existing_cached_artifact_binds_source_index_strictness(self):
        service_url = "https://example.invalid/MapServer"
        source_index_url = "https://example.invalid/source-index"
        reference = SimpleNamespace(dataset="tiny")
        region_reference = SimpleNamespace(
            dataset="tiny", manifest_digest="a" * 64)
        base_config = {
            "region_manifest_digest": "a" * 64,
            "provider": {
                "type": "arcgis_cached_map_service",
                "service_url": service_url,
            },
            "source_index_url": source_index_url,
            "jpeg_quality": satellite.DEFAULT_JPEG_QUALITY,
            "assembly_version": satellite.ASSEMBLY_VERSION,
        }

        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "existing"
            destination.mkdir()
            with mock.patch.object(
                    satellite.artifact, "open_artifact",
                    return_value=reference), mock.patch.object(
                        satellite.artifact, "load_manifest") as load_manifest:
                load_manifest.return_value = SimpleNamespace(config={
                    **base_config,
                    "source_index_coverage_required": False,
                })
                self.assertIs(
                    satellite._existing_artifact(
                        destination, region_reference,
                        service_url=service_url,
                        source_index_url=source_index_url,
                        require_source_index_coverage=False,
                        jpeg_quality=satellite.DEFAULT_JPEG_QUALITY),
                    reference)
                with self.assertRaisesRegex(
                        satellite.SatelliteError, "differs from request"):
                    satellite._existing_artifact(
                        destination, region_reference,
                        service_url=service_url,
                        source_index_url=source_index_url,
                        require_source_index_coverage=True,
                        jpeg_quality=satellite.DEFAULT_JPEG_QUALITY)

                # Historical strict artifacts omitted this field; absence
                # deliberately means strict coverage, not permissive reuse.
                load_manifest.return_value = SimpleNamespace(
                    config=base_config)
                self.assertIs(
                    satellite._existing_artifact(
                        destination, region_reference,
                        service_url=service_url,
                        source_index_url=source_index_url,
                        require_source_index_coverage=True,
                        jpeg_quality=satellite.DEFAULT_JPEG_QUALITY),
                    reference)
                with self.assertRaisesRegex(
                        satellite.SatelliteError, "differs from request"):
                    satellite._existing_artifact(
                        destination, region_reference,
                        service_url=service_url,
                        source_index_url=source_index_url,
                        require_source_index_coverage=False,
                        jpeg_quality=satellite.DEFAULT_JPEG_QUALITY)

    def test_source_tile_cache_resumes_and_replaces_corruption(self):
        grid = _tiny_grid()
        client = FakeTileClient(delay=0.002)
        with TemporaryDirectory() as temporary:
            build_dir = Path(temporary)
            first = satellite.ensure_source_tiles(
                build_dir, grid, client, workers=3)
            self.assertEqual(first["downloaded"], grid["n_source_tiles"])
            self.assertEqual(first["resumed"], 0)
            self.assertLessEqual(client.max_active_fetches, 3)
            satellite.write_source_tile_manifest(
                build_dir, grid, {"source": "fake"})

            calls_after_first = len(client.fetch_calls)
            second = satellite.ensure_source_tiles(
                build_dir, grid, client, workers=3)
            self.assertEqual(second["downloaded"], 0)
            self.assertEqual(second["resumed"], grid["n_source_tiles"])
            self.assertEqual(len(client.fetch_calls), calls_after_first)

            source_images = [
                path for path in build_dir.rglob("*")
                if path.is_file()
                and satellite.validate_image_file(path, (256, 256)) is not None
            ]
            self.assertEqual(len(source_images), grid["n_source_tiles"])
            victim = source_images[len(source_images) // 2]
            victim.write_bytes(b"truncated JPEG")

            third = satellite.ensure_source_tiles(
                build_dir, grid, client, workers=3)
            self.assertEqual(third["replaced"], 1)
            self.assertEqual(len(client.fetch_calls), calls_after_first + 1)
            self.assertIsNotNone(
                satellite.validate_image_file(victim, (256, 256)))

            victim.write_bytes(_jpeg_bytes((250, 1, 1)))
            fourth = satellite.ensure_source_tiles(
                build_dir, grid, client, workers=3)
            self.assertEqual(fourth["replaced"], 1)
            self.assertEqual(len(client.fetch_calls), calls_after_first + 2)

    def test_image_server_chunk_receipts_resume_and_rebuild_atomically(self):
        grid = _grid_with_source_shape(5, 3)
        x_min, y_min, _, _ = grid["source_tile_range_xyxy"]
        provider = {"source": "fixed-fake-image-server"}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_dir = root / "clean"
            client = FakeChunkImageServerClient(delay=0.002)
            first = satellite.ensure_source_tiles(
                clean_dir, grid, client, workers=8,
                image_server_chunk_tiles=3)
            self.assertEqual(first, {
                "total": 15, "resumed": 0,
                "downloaded": 15, "replaced": 0,
            })
            self.assertCountEqual(client.chunk_calls, [
                (grid["zoom"], x_min, y_min, 3, 3),
                (grid["zoom"], x_min + 3, y_min, 2, 3),
            ])
            self.assertLessEqual(client.max_active_fetches, 2)
            receipts = sorted(
                (clean_dir / "source_tile_chunks").rglob("*.json"))
            self.assertEqual(len(receipts), 2)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt["schema"],
                             satellite.IMAGE_SERVER_CHUNK_SCHEMA)
            self.assertEqual(receipt["chunking"]["shape_tiles_xy"], [3, 3])
            self.assertEqual(len(receipt["response"]["sha256"]), 64)
            self.assertEqual(
                len(receipt["response"]["decoded_pixel_sha256"]), 64)
            self.assertEqual(len(receipt["tiles"]), 9)

            clean_manifest = satellite.write_source_tile_manifest(
                clean_dir, grid, provider)
            clean_manifest_bytes = clean_manifest["path"].read_bytes()
            calls_after_first = len(client.chunk_calls)
            resumed = satellite.ensure_source_tiles(
                clean_dir, grid, client, workers=8,
                image_server_chunk_tiles=3)
            self.assertEqual(resumed["resumed"], 15)
            self.assertEqual(len(client.chunk_calls), calls_after_first)

            victim = satellite._tile_cache_path(
                clean_dir, grid["zoom"], x_min + 1, y_min + 1)
            victim.write_bytes(b"corrupt")
            repaired = satellite.ensure_source_tiles(
                clean_dir, grid, client, workers=8,
                image_server_chunk_tiles=3)
            self.assertEqual(repaired, {
                "total": 15, "resumed": 6,
                "downloaded": 0, "replaced": 9,
            })
            self.assertEqual(
                client.chunk_calls[-1],
                (grid["zoom"], x_min, y_min, 3, 3))
            for child_y in range(y_min, y_min + 3):
                for child_x in range(x_min, x_min + 3):
                    path = satellite._tile_cache_path(
                        clean_dir, grid["zoom"], child_x, child_y)
                    with Image.open(path) as child:
                        child.load()
                        self.assertEqual(
                            child.getpixel((128, 128)),
                            client.colour(child_x, child_y))

            # Even when all child images remain valid, an invalid commit
            # receipt cannot authorize mixing or partial reuse.
            first_receipt = satellite._source_chunk_receipt_path(
                clean_dir, grid["zoom"], x_min, y_min, 3, 3)
            first_receipt.write_text("{malformed", encoding="utf-8")
            calls_before_malformed = len(client.chunk_calls)
            malformed = satellite.ensure_source_tiles(
                clean_dir, grid, client, workers=8,
                image_server_chunk_tiles=3)
            self.assertEqual(malformed, {
                "total": 15, "resumed": 6,
                "downloaded": 0, "replaced": 9,
            })
            self.assertEqual(
                len(client.chunk_calls), calls_before_malformed + 1)
            self.assertEqual(
                client.chunk_calls[-1],
                (grid["zoom"], x_min, y_min, 3, 3))

            wrong_schema = json.loads(first_receipt.read_text())
            wrong_schema["schema"] = "wrong/v1"
            first_receipt.write_text(
                json.dumps(wrong_schema), encoding="utf-8")
            calls_before_schema = len(client.chunk_calls)
            invalid_schema = satellite.ensure_source_tiles(
                clean_dir, grid, client, workers=8,
                image_server_chunk_tiles=3)
            self.assertEqual(invalid_schema, {
                "total": 15, "resumed": 6,
                "downloaded": 0, "replaced": 9,
            })
            self.assertEqual(
                len(client.chunk_calls), calls_before_schema + 1)
            self.assertEqual(
                client.chunk_calls[-1],
                (grid["zoom"], x_min, y_min, 3, 3))

            # Valid-looking legacy single exports cannot be mixed into a
            # chunk recipe: without an atomic receipt, every child of the
            # fixed partition is replaced from that partition's response.
            partial_dir = root / "legacy-partial"
            for offset in (0, 2, 4):
                child_x = x_min + offset
                path = satellite._tile_cache_path(
                    partial_dir, grid["zoom"], child_x, y_min)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_png_bytes((250, 1, 1)))
            partial_client = FakeChunkImageServerClient()
            partial = satellite.ensure_source_tiles(
                partial_dir, grid, partial_client, workers=8,
                image_server_chunk_tiles=3)
            self.assertEqual(partial["total"], 15)
            self.assertEqual(partial["replaced"], 3)
            self.assertEqual(partial["downloaded"], 12)
            self.assertCountEqual(partial_client.chunk_calls, [
                (grid["zoom"], x_min, y_min, 3, 3),
                (grid["zoom"], x_min + 3, y_min, 2, 3),
            ])
            partial_manifest = satellite.write_source_tile_manifest(
                partial_dir, grid, provider)
            self.assertEqual(
                partial_manifest["path"].read_bytes(),
                clean_manifest_bytes)

    def test_patch_cache_is_row_major_resumable_and_repairs_corruption(self):
        grid = _tiny_grid()
        client = FakeTileClient()
        with TemporaryDirectory() as temporary:
            build_dir = Path(temporary)
            satellite.ensure_source_tiles(build_dir, grid, client, workers=2)
            source_manifest = satellite.write_source_tile_manifest(
                build_dir, grid, {"source": "fake"})
            first = satellite.ensure_patches(
                build_dir, grid, source_manifest["sha256"],
                jpeg_quality=91, workers=2)
            self.assertEqual(first["written"], grid["n_patches"])

            patch_dir = Path(first["patch_dir"])
            patch_manifest = satellite.write_patch_manifest(
                build_dir, grid, source_manifest["sha256"],
                jpeg_quality=91)
            metadata = json.loads(
                Path(patch_manifest["manifest_path"]).read_text())
            rows = metadata["patches"]
            self.assertEqual([row["index"] for row in rows], [0, 1, 2, 3])
            self.assertEqual(
                [(row["row"], row["column"]) for row in rows],
                [(0, 0), (0, 1), (1, 0), (1, 1)])
            for row in rows:
                self.assertRegex(
                    row["filename"],
                    r"^satellite_-?\d+\.\d{8}_-?\d+\.\d{8}\.jpg$")
                self.assertTrue((patch_dir / row["filename"]).is_file())

            mtimes = {
                path.name: path.stat().st_mtime_ns
                for path in patch_dir.glob("*.jpg")
            }
            second = satellite.ensure_patches(
                build_dir, grid, source_manifest["sha256"],
                jpeg_quality=91, workers=2)
            self.assertEqual(second["written"], 0)
            self.assertEqual(second["resumed"], grid["n_patches"])

            victim = sorted(patch_dir.glob("*.jpg"))[1]
            victim.write_bytes(b"corrupt patch")
            third = satellite.ensure_patches(
                build_dir, grid, source_manifest["sha256"],
                jpeg_quality=91, workers=2)
            self.assertEqual(third["replaced"], 1)
            self.assertIsNotNone(
                satellite.validate_image_file(victim, (640, 640)))
            self.assertTrue(any(
                path.stat().st_mtime_ns == mtimes[path.name]
                for path in patch_dir.glob("*.jpg") if path != victim))

            # A decodable but wrong/stale JPEG must not bypass a complete
            # prior manifest merely because its dimensions are valid.
            stale = sorted(patch_dir.glob("*.jpg"))[2]
            stale.write_bytes(_jpeg_bytes((250, 1, 1)))
            fourth = satellite.ensure_patches(
                build_dir, grid, source_manifest["sha256"],
                jpeg_quality=91, workers=2)
            self.assertEqual(fourth["replaced"], 1)

    def test_build_cache_can_reuse_sources_across_assembly_versions(self):
        with TemporaryDirectory() as temporary:
            build_dir = Path(temporary) / "build"
            state = {
                "schema": satellite.BUILD_SCHEMA,
                "dataset": "tiny",
                "version": "cache_v1",
                "region_manifest_digest": "a" * 64,
                "region_content_digest": "c" * 64,
                "assembly_version": "old",
            }
            satellite._prepare_build_directory(build_dir, state)
            changed_assembly = {**state, "assembly_version": "new"}
            satellite._prepare_build_directory(
                build_dir, changed_assembly)
            with self.assertRaisesRegex(
                    satellite.SatelliteError, "different recipe"):
                satellite._prepare_build_directory(
                    build_dir,
                    {**changed_assembly,
                     "region_content_digest": "d" * 64})

    def test_missing_source_tile_cannot_produce_a_source_manifest(self):
        grid = _tiny_grid()
        x_min, y_min, _, _ = grid["source_tile_range_xyxy"]
        client = FakeTileClient(missing_tiles={(x_min, y_min)})
        with TemporaryDirectory() as temporary:
            build_dir = Path(temporary)
            with self.assertRaisesRegex(satellite.SatelliteError, "missing"):
                satellite.ensure_source_tiles(
                    build_dir, grid, client, workers=2)
            with self.assertRaises(satellite.SatelliteError):
                satellite.write_source_tile_manifest(
                    build_dir, grid, {"source": "fake"})
            self.assertFalse(list(build_dir.rglob("*manifest*.json")))

    def test_materialize_never_publishes_after_a_missing_source_tile(self):
        grid = _tiny_grid()
        x_min, y_min, _, _ = grid["source_tile_range_xyxy"]
        client = FakeTileClient(missing_tiles={(x_min, y_min)})
        region_reference = SimpleNamespace(
            dataset="tiny",
            manifest_digest="a" * 64,
            content_digest="b" * 64,
        )
        plan = {"grid": grid}
        with TemporaryDirectory() as temporary:
            farfield_root = Path(temporary)
            destination = (
                farfield_root / "artifacts" / satellite.ARTIFACT_KIND
                / "tiny" / "massgis_v1")
            with mock.patch.object(
                    satellite.region, "load_region",
                    return_value=(region_reference, plan)):
                with self.assertRaisesRegex(
                        satellite.SatelliteError, "missing"):
                    satellite.materialize(
                        farfield_root=farfield_root, dataset="tiny",
                        region_dir=farfield_root / "fake-region",
                        version="massgis_v1", source_index_url=None,
                        workers=2, patch_workers=2, client=client)
            self.assertFalse(destination.exists())
            self.assertFalse(
                destination.with_name(destination.name + ".incomplete")
                .exists())


if __name__ == "__main__":
    unittest.main()
