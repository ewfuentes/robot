"""Small, dependency-free decoder for GPS-bearing Garmin FIT messages.

This intentionally implements only the FIT message and field types needed by
the self-collection import path.  Unknown messages and fields are preserved by
number or skipped according to their declared size; malformed input fails
loudly.  It is based on the decoder retained with the 2026-08-15 Mount
Washington collection, moved here so future collections do not need to copy a
one-off parser into ``raw_material``.
"""

import datetime
import struct
from pathlib import Path


BASE_TYPES = {
    0x00: ("enum", "B", 1, 0xFF),
    0x01: ("sint8", "b", 1, 0x7F),
    0x02: ("uint8", "B", 1, 0xFF),
    0x83: ("sint16", "h", 2, 0x7FFF),
    0x84: ("uint16", "H", 2, 0xFFFF),
    0x85: ("sint32", "i", 4, 0x7FFFFFFF),
    0x86: ("uint32", "I", 4, 0xFFFFFFFF),
    0x07: ("string", "s", 1, 0x00),
    0x88: ("float32", "f", 4, None),
    0x89: ("float64", "d", 8, None),
    0x0A: ("uint8z", "B", 1, 0x00),
    0x8B: ("uint16z", "H", 2, 0x0000),
    0x8C: ("uint32z", "I", 4, 0x00000000),
    0x0D: ("byte", "B", 1, 0xFF),
    0x8E: ("sint64", "q", 8, 0x7FFFFFFFFFFFFFFF),
    0x8F: ("uint64", "Q", 8, 0xFFFFFFFFFFFFFFFF),
    0x90: ("uint64z", "Q", 8, 0),
}

MESSAGE_NAMES = {
    0: "file_id",
    18: "session",
    19: "lap",
    20: "record",
    21: "event",
    23: "device_info",
    34: "activity",
    49: "file_creator",
}

RECORD_FIELDS = {
    253: "timestamp",
    0: "position_lat",
    1: "position_long",
    2: "altitude",
    3: "heart_rate",
    4: "cadence",
    5: "distance",
    6: "speed",
    13: "temperature",
    41: "gps_accuracy",
    73: "enhanced_speed",
    78: "enhanced_altitude",
}
SESSION_FIELDS = {
    253: "timestamp",
    2: "start_time",
    5: "sport",
    6: "sub_sport",
    7: "total_elapsed_time",
    8: "total_timer_time",
    9: "total_distance",
    11: "total_calories",
    22: "total_ascent",
    23: "total_descent",
    254: "message_index",
}
EVENT_FIELDS = {
    253: "timestamp",
    0: "event",
    1: "event_type",
    3: "data",
    4: "event_group",
}
LAP_FIELDS = {
    253: "timestamp",
    2: "start_time",
    7: "total_elapsed_time",
    8: "total_timer_time",
    9: "total_distance",
    254: "message_index",
}
ACTIVITY_FIELDS = {
    253: "timestamp",
    0: "total_timer_time",
    1: "num_sessions",
    2: "type",
    3: "event",
    4: "event_type",
    5: "local_timestamp",
}
FILE_ID_FIELDS = {
    0: "type",
    1: "manufacturer",
    2: "product",
    3: "serial_number",
    4: "time_created",
    5: "number",
    8: "product_name",
}
DEVICE_FIELDS = {
    253: "timestamp",
    0: "device_index",
    1: "device_type",
    2: "manufacturer",
    3: "serial_number",
    4: "product",
    5: "software_version",
    27: "product_name",
}
FIELD_MAPS = {
    "record": RECORD_FIELDS,
    "session": SESSION_FIELDS,
    "event": EVENT_FIELDS,
    "lap": LAP_FIELDS,
    "activity": ACTIVITY_FIELDS,
    "file_id": FILE_ID_FIELDS,
    "device_info": DEVICE_FIELDS,
}

FIT_EPOCH = datetime.datetime(1989, 12, 31, tzinfo=datetime.timezone.utc)
SEMICIRCLE_DEG = 180.0 / 2**31
ALTITUDE_SCALE = 5.0
ALTITUDE_OFFSET = 500.0
SPEED_SCALE = 1000.0
DISTANCE_SCALE = 100.0
TIME_SCALE = 1000.0


def decode(path: Path | str) -> list[tuple[str, dict]]:
    """Return decoded ``(message_name, fields)`` pairs in file order."""
    data = Path(path).read_bytes()
    if len(data) < 12 or data[8:12] != b".FIT":
        raise ValueError(f"not a FIT file: {path}")
    header_size = data[0]
    if header_size < 12 or len(data) < header_size:
        raise ValueError(f"invalid FIT header size {header_size}: {path}")
    _, _, data_size = struct.unpack("<BHI", data[1:8])
    pos, end = header_size, header_size + data_size
    if end > len(data):
        raise ValueError(f"truncated FIT data section: {path}")

    definitions: dict[int, dict] = {}
    output: list[tuple[str, dict]] = []
    while pos < end:
        header = data[pos]
        pos += 1
        if header & 0x80:
            local = (header >> 5) & 0x3
            if definitions.get(local) is None:
                raise ValueError(
                    "compressed timestamp for undefined local message")
            # The self-collection files seen to date do not use compressed
            # timestamp headers. Their five-bit offset replaces timestamp
            # bytes in the data message, so decoding the normal definition
            # first would consume the following fields at the wrong offsets.
            # Reject before reading any record bytes until reconstruction is
            # implemented with the required previous-timestamp state.
            raise ValueError(
                "compressed FIT timestamps are not supported by this "
                "minimal decoder")

        local = header & 0x0F
        if header & 0x40:
            pos += 1  # reserved
            byte_order = data[pos]
            if byte_order not in (0, 1):
                raise ValueError(f"invalid FIT byte order {byte_order}")
            fmt = "<" if byte_order == 0 else ">"
            pos += 1
            global_number = struct.unpack(fmt + "H", data[pos:pos + 2])[0]
            pos += 2
            field_count = data[pos]
            pos += 1
            fields = []
            for _ in range(field_count):
                fields.append((data[pos], data[pos + 1], data[pos + 2]))
                pos += 3
            developer_fields = []
            if header & 0x20:
                developer_count = data[pos]
                pos += 1
                for _ in range(developer_count):
                    developer_fields.append(
                        (data[pos], data[pos + 1], data[pos + 2]))
                    pos += 3
            definitions[local] = {
                "name": MESSAGE_NAMES.get(
                    global_number, f"msg{global_number}"),
                "fmt": fmt,
                "fields": fields,
                "developer_fields": developer_fields,
            }
        else:
            definition = definitions.get(local)
            if definition is None:
                raise ValueError(f"data for undefined local message {local}")
            values, pos = _read_data(data, pos, definition)
            output.append((definition["name"], values))
    return output


def _read_data(data: bytes, pos: int, definition: dict) -> tuple[dict, int]:
    fmt = definition["fmt"]
    names = FIELD_MAPS.get(definition["name"], {})
    values = {}
    for field_number, size, base_type in definition["fields"]:
        if pos + size > len(data):
            raise ValueError("truncated FIT message")
        spec = BASE_TYPES.get(base_type)
        if spec is None:
            pos += size
            continue
        type_name, code, unit_size, invalid = spec
        if size % unit_size:
            raise ValueError(
                f"FIT field size {size} is not a multiple of {unit_size}")
        if type_name == "string":
            value = data[pos:pos + size].split(b"\x00")[0].decode(
                "utf-8", "replace")
        else:
            count = size // unit_size
            items = struct.unpack(
                fmt + code * count, data[pos:pos + size])
            clean = [
                None if invalid is not None and item == invalid else item
                for item in items
            ]
            value = clean[0] if count == 1 else clean
        pos += size
        values[names.get(field_number, field_number)] = value
    # A developer-field definition stores a developer-data index in its third
    # byte, not a FIT base type.  Resolving it requires the corresponding
    # field-description message.  These fields are not needed for GPS import,
    # so preserve framing by skipping their declared bytes instead of risking
    # a plausible but incorrectly typed value.
    for _, size, _ in definition["developer_fields"]:
        if pos + size > len(data):
            raise ValueError("truncated FIT developer field")
        pos += size
    return values, pos


def timestamp(value: int | None) -> datetime.datetime | None:
    """Convert FIT epoch seconds to an aware UTC datetime."""
    if value is None:
        return None
    return FIT_EPOCH + datetime.timedelta(seconds=value)
