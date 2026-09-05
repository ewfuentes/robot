#!/usr/bin/env python3
"""Tests for the Mapbox Vector Tile decoder and tile arithmetic.

    bazel test //experimental/overhead_matching/swag/farfield/collection:vector_tiles_test

No network and no token: the decoder is fed protobuf built here, so a failure
means the decoder is wrong rather than that Mapillary changed. The one test that
does hit the network is skipped unless MLY_LIVE=1.
"""

import math
import os
import struct
import unittest

from experimental.overhead_matching.swag.farfield.collection import vector_tiles as vt


# --------------------------------------------------------------------------
# minimal MVT encoder, so tests assert against bytes we constructed
# --------------------------------------------------------------------------

def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _tag(field_no: int, wire: int) -> bytes:
    return _varint((field_no << 3) | wire)


def _length_delimited(field_no: int, payload: bytes) -> bytes:
    return _tag(field_no, 2) + _varint(len(payload)) + payload


def _zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 63) if n < 0 else n << 1


def encode_value(value) -> bytes:
    if isinstance(value, bool):
        return _length_delimited(4, _tag(7, 0) + _varint(1 if value else 0))
    if isinstance(value, str):
        return _length_delimited(4, _length_delimited(1, value.encode()))
    if isinstance(value, int):
        return _length_delimited(4, _tag(4, 0) + _varint(value))
    if isinstance(value, float):
        return _length_delimited(4, _tag(3, 1) + struct.pack("<d", value))
    raise TypeError(type(value))


def encode_tile(layer_name: str, features: list, extent: int = 4096) -> bytes:
    """Build a one-layer tile. `features` is [(geom_type, [(x, y), ...], props)]."""
    keys, values = [], []
    key_index, value_index = {}, {}

    def intern(dictionary, store, item):
        marker = (type(item).__name__, item)
        if marker not in dictionary:
            dictionary[marker] = len(store)
            store.append(item)
        return dictionary[marker]

    encoded_features = []
    for geom_type, points, props in features:
        tags = []
        for key, value in props.items():
            if key not in key_index:
                key_index[key] = len(keys)
                keys.append(key)
            tags.append(key_index[key])
            tags.append(intern(value_index, values, value))

        geometry = []
        cursor_x = cursor_y = 0
        if geom_type == vt.GEOM_POINT:
            geometry.append((1 << 3) | 1)  # MoveTo, count 1
            for x, y in points[:1]:
                geometry += [_zigzag_encode(x - cursor_x), _zigzag_encode(y - cursor_y)]
                cursor_x, cursor_y = x, y
        else:
            geometry.append((1 << 3) | 1)
            x, y = points[0]
            geometry += [_zigzag_encode(x), _zigzag_encode(y)]
            cursor_x, cursor_y = x, y
            geometry.append(((len(points) - 1) << 3) | 2)  # LineTo
            for x, y in points[1:]:
                geometry += [_zigzag_encode(x - cursor_x), _zigzag_encode(y - cursor_y)]
                cursor_x, cursor_y = x, y

        body = b""
        if tags:
            body += _length_delimited(2, b"".join(_varint(t) for t in tags))
        body += _tag(3, 0) + _varint(geom_type)
        body += _length_delimited(4, b"".join(_varint(g) for g in geometry))
        encoded_features.append(_length_delimited(2, body))

    layer = _length_delimited(1, layer_name.encode())
    layer += b"".join(encoded_features)
    layer += b"".join(_length_delimited(3, k.encode()) for k in keys)
    layer += b"".join(encode_value(v) for v in values)
    layer += _tag(5, 0) + _varint(extent)
    layer += _tag(15, 0) + _varint(2)
    return _length_delimited(3, layer)


# --------------------------------------------------------------------------

class VarintTest(unittest.TestCase):
    def test_round_trip(self):
        for value in (0, 1, 127, 128, 300, 16384, 2 ** 31, 2 ** 53):
            decoded, pos = vt._read_varint(_varint(value), 0)
            self.assertEqual(decoded, value)
            self.assertEqual(pos, len(_varint(value)))

    def test_zigzag_decodes_negatives(self):
        for value in (-2 ** 20, -300, -1, 0, 1, 300, 2 ** 20):
            self.assertEqual(vt._zigzag(_zigzag_encode(value)), value)


class TileArithmeticTest(unittest.TestCase):
    def test_point_falls_inside_the_tile_it_maps_to(self):
        # Asserted as an invariant rather than a magic index: the round trip
        # through tile_bounds is the property that actually matters, and a
        # hand-computed y is just an opportunity to get Mercator wrong.
        lon, lat, z = 6.6323, 46.5197, 14
        x, y = vt.lonlat_to_tile(lon, lat, z)
        west, south, east, north = vt.tile_bounds(z, x, y)
        self.assertTrue(west <= lon <= east)
        self.assertTrue(south <= lat <= north)

    def test_origin_is_top_left(self):
        # Tile (0,0) at any zoom is the north-west corner of the world.
        self.assertEqual(vt.lonlat_to_tile(-179.9, 85.0, 4), (0, 0))

    def test_bounds_round_trip(self):
        z, x, y = 12, 2130, 1449
        west, south, east, north = vt.tile_bounds(z, x, y)
        self.assertLess(west, east)
        self.assertLess(south, north)
        centre_lon = (west + east) / 2
        centre_lat = (south + north) / 2
        self.assertEqual(vt.lonlat_to_tile(centre_lon, centre_lat, z), (x, y))

    def test_latitude_clamps_at_mercator_limit(self):
        n = 2 ** 6
        _x, y = vt.lonlat_to_tile(0.0, 89.9, 6)
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, n)

    def test_tiles_for_bbox_covers_corners(self):
        bbox = (6.10, 46.30, 7.00, 46.60)
        tiles = vt.tiles_for_bbox(bbox, 12)
        self.assertIn(vt.lonlat_to_tile(bbox[0], bbox[3], 12), tiles)
        self.assertIn(vt.lonlat_to_tile(bbox[2], bbox[1], 12), tiles)


class GeometryDecodeTest(unittest.TestCase):
    def test_cursor_persists_across_commands(self):
        # LineTo deltas are relative to where MoveTo left the cursor, not to
        # the tile origin. Resetting between commands is the classic MVT bug.
        commands = [(1 << 3) | 1, _zigzag_encode(100), _zigzag_encode(200),
                    (2 << 3) | 2,
                    _zigzag_encode(10), _zigzag_encode(0),
                    _zigzag_encode(0), _zigzag_encode(-20)]
        rings = vt._decode_geometry(commands, vt.GEOM_LINESTRING)
        self.assertEqual(rings, [[(100, 200), (110, 200), (110, 180)]])

    def test_second_moveto_starts_a_new_ring(self):
        # Command integer packs id in the low 3 bits and count above:
        # (count << 3) | id. MoveTo is id 1, so two MoveTos is (2 << 3) | 1.
        commands = [(1 << 3) | 1, _zigzag_encode(5), _zigzag_encode(5),
                    (2 << 3) | 1,
                    _zigzag_encode(1), _zigzag_encode(1),
                    _zigzag_encode(50), _zigzag_encode(0)]
        rings = vt._decode_geometry(commands, vt.GEOM_POINT)
        self.assertEqual(len(rings), 3)

    def test_unknown_command_raises(self):
        with self.assertRaises(ValueError):
            vt._decode_geometry([(1 << 3) | 5], vt.GEOM_POINT)


class DecodeTileTest(unittest.TestCase):
    def test_point_lands_inside_its_tile(self):
        z, x, y = 12, 2130, 1449
        blob = encode_tile("image", [(vt.GEOM_POINT, [(2048, 2048)],
                                      {"id": 1234, "is_pano": True})])
        layers = vt.decode_tile(blob, (z, x, y))

        feature, = layers["image"]
        west, south, east, north = vt.tile_bounds(z, x, y)
        lon, lat = feature.coords[0]
        self.assertTrue(west <= lon <= east)
        self.assertTrue(south <= lat <= north)
        # 2048/4096 is the tile centre in x; y is non-linear in Mercator, so
        # only longitude is pinned exactly.
        self.assertAlmostEqual(lon, (west + east) / 2, places=9)

    def test_properties_decode_by_type(self):
        blob = encode_tile("sequence", [(vt.GEOM_POINT, [(0, 0)], {
            "id": "abc123", "captured_at": 1560423801000,
            "is_pano": True, "foot": False, "quality_score": 0.7158,
        })])
        props = vt.decode_tile(blob)["sequence"][0].properties

        self.assertEqual(props["id"], "abc123")
        self.assertEqual(props["captured_at"], 1560423801000)
        self.assertIs(props["is_pano"], True)
        self.assertIs(props["foot"], False)
        self.assertAlmostEqual(props["quality_score"], 0.7158, places=6)

    def test_bool_false_is_not_confused_with_absent(self):
        blob = encode_tile("sequence", [(vt.GEOM_POINT, [(0, 0)], {"is_pano": False})])
        props = vt.decode_tile(blob)["sequence"][0].properties
        self.assertIn("is_pano", props)
        self.assertIs(props["is_pano"], False)

    def test_non_default_extent_is_honoured(self):
        z, x, y = 10, 100, 100
        west, _s, east, _n = vt.tile_bounds(z, x, y)
        blob = encode_tile("l", [(vt.GEOM_POINT, [(512, 512)], {})], extent=1024)
        lon, _lat = vt.decode_tile(blob, (z, x, y))["l"][0].coords[0]
        self.assertAlmostEqual(lon, (west + east) / 2, places=9)

    def test_gzipped_input_is_sniffed(self):
        import gzip
        blob = encode_tile("l", [(vt.GEOM_POINT, [(1, 1)], {"id": "x"})])
        self.assertEqual(vt.decode_tile(gzip.compress(blob))["l"][0].properties["id"], "x")

    def test_sequence_id_reads_either_layer_convention(self):
        image = vt.TileFeature("image", 1, {"sequence_id": "seq1", "id": 99}, [])
        sequence = vt.TileFeature("sequence", 2, {"id": "seq2"}, [])
        self.assertEqual(image.sequence_id, "seq1")
        self.assertEqual(sequence.sequence_id, "seq2")


class MergeTest(unittest.TestCase):
    @staticmethod
    def _fragment(seq_id, coords):
        return vt.TileFeature("sequence", vt.GEOM_LINESTRING,
                              {"id": seq_id, "is_pano": True}, coords)

    def test_fragments_of_one_sequence_join(self):
        merged = vt.merge_tile_features([
            self._fragment("s", [(0.0, 0.0), (0.1, 0.0)]),
            self._fragment("s", [(0.1, 0.0), (0.2, 0.0)]),
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged["s"]["n_fragments"], 2)
        self.assertGreater(merged["s"]["length_km"], 20)

    def test_fragment_order_is_spatial_not_fetch_order(self):
        # Tiles arrive row-major, so a diagonal track's fragments come back
        # scrambled; joining in arrival order zig-zags and inflates length.
        ordered = vt.merge_tile_features([
            self._fragment("s", [(0.0, 0.0), (0.1, 0.0)]),
            self._fragment("s", [(0.1, 0.0), (0.2, 0.0)]),
            self._fragment("s", [(0.2, 0.0), (0.3, 0.0)]),
        ])["s"]["length_km"]
        scrambled = vt.merge_tile_features([
            self._fragment("s", [(0.2, 0.0), (0.3, 0.0)]),
            self._fragment("s", [(0.0, 0.0), (0.1, 0.0)]),
            self._fragment("s", [(0.1, 0.0), (0.2, 0.0)]),
        ])["s"]["length_km"]
        self.assertAlmostEqual(ordered, scrambled, delta=0.5)

    def test_reversed_fragment_is_flipped_not_doubled_back(self):
        joined = vt.merge_tile_features([
            self._fragment("s", [(0.0, 0.0), (0.1, 0.0)]),
            self._fragment("s", [(0.2, 0.0), (0.1, 0.0)]),   # runs backwards
        ])["s"]
        self.assertAlmostEqual(joined["length_km"],
                               vt.polyline_length_km([(0.0, 0.0), (0.2, 0.0)]),
                               delta=0.5)

    def test_distinct_sequences_stay_separate(self):
        merged = vt.merge_tile_features([
            self._fragment("a", [(0.0, 0.0), (0.1, 0.0)]),
            self._fragment("b", [(1.0, 1.0), (1.1, 1.0)]),
        ])
        self.assertEqual(set(merged), {"a", "b"})

    def test_features_without_an_id_are_skipped(self):
        nameless = vt.TileFeature("sequence", vt.GEOM_LINESTRING, {}, [(0.0, 0.0)])
        self.assertEqual(vt.merge_tile_features([nameless]), {})


class LengthTest(unittest.TestCase):
    def test_one_degree_of_latitude(self):
        self.assertAlmostEqual(
            vt.polyline_length_km([(0.0, 0.0), (0.0, 1.0)]), 110.6, delta=0.5)

    def test_longitude_shrinks_with_latitude(self):
        at_equator = vt.polyline_length_km([(0.0, 0.0), (1.0, 0.0)])
        at_sixty = vt.polyline_length_km([(0.0, 60.0), (1.0, 60.0)])
        self.assertAlmostEqual(at_sixty / at_equator, 0.5, delta=0.02)

    def test_single_point_has_no_length(self):
        self.assertEqual(vt.polyline_length_km([(1.0, 1.0)]), 0.0)


@unittest.skipUnless(os.environ.get("MLY_LIVE") == "1", "set MLY_LIVE=1 to hit the API")
class LiveTest(unittest.TestCase):
    def test_lausanne_tile_has_the_expected_layers(self):
        client = VectorTileClient() if False else vt.VectorTileClient()
        x, y = vt.lonlat_to_tile(6.6323, 46.5197, 14)
        layers = client.fetch(14, x, y)
        self.assertIn("sequence", layers)
        self.assertIn("image", layers)
        feature = layers["sequence"][0]
        for field in ("id", "is_pano", "captured_at", "image_id"):
            self.assertIn(field, feature.properties)


if __name__ == "__main__":
    unittest.main(verbosity=2)
