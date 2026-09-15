import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield.dataset_tools import fit_decoder


class FitDecoderTest(unittest.TestCase):

    def test_compressed_timestamp_rejected_before_reading_record_bytes(self):
        # Define local message 0 as a record containing a normal timestamp,
        # then emit a compressed-timestamp header for that local message. FIT
        # omits the normal timestamp bytes after this header, so passing the
        # position to _read_data would immediately misframe the record.
        definition = bytes([
            0x40,  # normal definition header, local message 0
            0x00,  # reserved
            0x00,  # little-endian architecture
            20, 0,  # global record message
            1,  # one field
            253, 4, 0x86,  # timestamp: uint32
        ])
        payload = definition + bytes([0x81])
        header = struct.pack(
            "<BBHI4s", 12, 0x10, 0, len(payload), b".FIT")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compressed.fit"
            path.write_bytes(header + payload)
            with mock.patch.object(
                    fit_decoder, "_read_data",
                    side_effect=AssertionError("record bytes were consumed")) as reader:
                with self.assertRaisesRegex(
                        ValueError, "compressed FIT timestamps are not supported"):
                    fit_decoder.decode(path)
            reader.assert_not_called()

    def test_developer_field_bytes_are_skipped_without_guessing_their_type(self):
        values, position = fit_decoder._read_data(
            bytes([42, 0xAA, 0xBB]),
            0,
            {
                "name": "unknown_message",
                "fmt": "<",
                "fields": [(1, 1, 0x02)],
                # The third value is a developer-data index, deliberately 0;
                # treating it as base type 0 would create a bogus field 99.
                "developer_fields": [(99, 2, 0)],
            },
        )
        self.assertEqual(values, {1: 42})
        self.assertEqual(position, 3)

    def test_truncated_developer_field_fails(self):
        with self.assertRaisesRegex(ValueError, "truncated FIT developer field"):
            fit_decoder._read_data(
                b"\x01",
                0,
                {
                    "name": "unknown_message",
                    "fmt": "<",
                    "fields": [],
                    "developer_fields": [(99, 2, 0)],
                },
            )


if __name__ == "__main__":
    unittest.main()
