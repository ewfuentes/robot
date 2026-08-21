"""A collection of Argoverse 2 logs, addressed by log id.

One :class:`~av2_log.LogSource` per log and nothing more. Locating files, deciding which streams
a log has, and parsing them all belong to that class, so what is left here is only the index over
*many* logs -- which ones exist, and which of them this dataset covers.

Logs are addressed by id rather than by integer position, and there is deliberately no
``__getitem__``: a map-style torch dataset indexes *samples*, and a sample is not a log. Whatever
defines one -- a pose along the drive, a map crop, a sensor instant -- belongs to the module that
knows what it is training on, built over the sources this hands out.
"""

from pathlib import Path
from typing import Iterator

from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.data import av2_log


class ArgoverseDataset:
    """Logs of one dataset+split on disk, as lazily-read sources.

    Construction is cheap and does no parsing: a :class:`~av2_log.LogSource` is one ``stat`` of
    the log directory, and every stream it exposes is read on the call that asks for it. So
    building this over all 700 ``sensor/train`` logs costs 700 stats, not 700 map parses.

    Args:
        request: dataset+split the logs belong to. Its ``items`` selection is ignored -- what
            matters is what is on disk, which each source reports for itself.
        log_ids: which logs to cover. ``None`` takes every log present under `request`.
        root: local dataset root, mirroring S3.

    Raises:
        av2_log.MissingStreamError: if a log named in `log_ids` has no directory. Named logs are
            required rather than skipped -- the caller asked for those specifically, and
            silently dropping one turns a typo into a short dataset.
    """

    def __init__(
        self,
        request: al.Request,
        log_ids: list[str] | None = None,
        root: Path = al.DEFAULT_ROOT,
    ) -> None:
        ids = av2_log.discover_log_ids(request, root) if log_ids is None else log_ids
        self._logs = {log_id: av2_log.LogSource(request, log_id, root) for log_id in ids}

    def __len__(self) -> int:
        return len(self._logs)

    def get_log_ids(self) -> list[str]:
        """The ids this dataset covers, in the order they were given or discovered."""
        return list(self._logs)

    def log(self, log_id: str) -> av2_log.LogSource:
        """The source for one log."""
        return self._logs[log_id]

    def logs(self) -> Iterator[av2_log.LogSource]:
        """Every source, in the same order as :meth:`get_log_ids`."""
        return iter(self._logs.values())
