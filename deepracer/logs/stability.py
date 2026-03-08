"""Simtrace timing stability analysis."""

from __future__ import annotations

import csv
import os
import re
import warnings
from io import StringIO

import numpy as np
import pandas as pd

from .handler import FileHandler
from .misc import LogType

_WALL_CLOCK_CANDIDATES = ("wall_clock", "wallclock", "wall_time")
_MS_PER_SECOND = 1000.0
_ITERATION_RE = re.compile(r"^(?P<it>\d+)-iteration\.csv$")


def _basename(path: str) -> str:
    """Return the final component of a path, handling both '/' and os.sep."""
    name = path.rsplit("/", 1)[-1]
    if os.sep != "/":
        name = name.rsplit(os.sep, 1)[-1]
    return name


def _extract_iteration(path: str):
    """Extract the iteration integer from a simtrace file path, or ``None``."""
    m = _ITERATION_RE.match(_basename(path))
    return int(m.group("it")) if m else None


def _flatten(per_episode: dict) -> np.ndarray:
    chunks = [v for v in per_episode.values() if v.size > 0]
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)


def _summarize(values: np.ndarray) -> dict | None:
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "avg": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def parse_simtrace_bytes(data: bytes) -> tuple:
    """Parse a simtrace CSV byte payload into per-episode step deltas.

    All rows are included in the timestamp sequence regardless of
    ``episode_status``.  The single exception is the delta that *arrives*
    at a ``prepare`` row (the inter-episode reset gap), which is excluded.
    The delta *from* a ``prepare`` row to the next row is kept.

    If the CSV contains a ``wall_clock`` column the real-time factor (RTF)
    is also computed as simulated-seconds / wall-clock-seconds.

    Args:
        data:
            Raw bytes of a simtrace iteration CSV (may include a header row).

    Returns:
        A 2-tuple ``(per_episode_deltas, real_time_factor)`` where
        ``per_episode_deltas`` is a ``dict[int, np.ndarray]`` of non-negative
        step deltas **in seconds** and ``real_time_factor`` is a ``float`` or
        ``None`` when wall-clock data is unavailable.

    Raises:
        ValueError: If the CSV is missing required columns (``episode``,
            ``tstamp``, ``episode_status``).
    """
    text = data.decode("utf-8", errors="replace")
    reader = csv.DictReader(StringIO(text))
    fieldnames = reader.fieldnames or []
    missing = {"episode", "tstamp", "episode_status"} - set(fieldnames)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    lower_map = {n.lower(): n for n in fieldnames}
    wall_field = next((lower_map[c] for c in _WALL_CLOCK_CANDIDATES if c in lower_map), None)

    per_episode_data: dict = {}
    sim_wall_points = []

    for row in reader:
        ep = int(float(row["episode"]))
        ts = float(row["tstamp"])
        status = (row.get("episode_status") or "").strip().lower()
        per_episode_data.setdefault(ep, []).append((ts, status))

        if wall_field:
            raw = (row.get(wall_field) or "").strip()
            if raw:
                try:
                    sim_wall_points.append((ts, float(raw)))
                except ValueError:
                    pass

    per_episode_deltas = {}
    for ep, data in per_episode_data.items():
        if len(data) < 2:
            continue
        timestamps = np.array([d[0] for d in data], dtype=np.float64)
        statuses = [d[1] for d in data]
        deltas = np.diff(timestamps)
        # Drop the delta arriving AT a 'prepare' row (episode reset gap).
        keep = np.array([statuses[i + 1] != "prepare" for i in range(len(deltas))])
        deltas = deltas[keep]
        per_episode_deltas[ep] = deltas[deltas >= 0]

    sim_sum = wall_sum = 0.0
    for i in range(1, len(sim_wall_points)):
        ps, pw = sim_wall_points[i - 1]
        cs, cw = sim_wall_points[i]
        ds, dw = cs - ps, cw - pw
        if ds >= 0 and dw > 0:
            sim_sum += ds
            wall_sum += dw
    rtf = (sim_sum / wall_sum) if wall_sum > 0 else None

    return per_episode_deltas, rtf


class SimtraceStabilityAnalyzer:
    """Analyses simtrace timing stability using a :class:`~deepracer.logs.FileHandler`.

    Computes per-iteration step-delta statistics from the ``tstamp`` column of
    simtrace CSV files.  All row statuses are included except for the reset gap
    that arrives *at* a ``prepare`` row.

    Example::

        from deepracer.logs import DeepRacerLog, LogType

        log = DeepRacerLog("./my-model")
        df = log.stability.analyze()                          # training
        df_eval = log.stability.analyze(LogType.EVALUATION)  # evaluation

    Alternatively::

        from deepracer.logs import FSFileHandler, SimtraceStabilityAnalyzer

        fh = FSFileHandler("./my-model")
        fh.determine_root_folder_type()
        analyzer = SimtraceStabilityAnalyzer(fh)
        df = analyzer.analyze()

    """

    def __init__(self, filehandler: FileHandler):
        self._fh = filehandler

    def analyze(self, log_type: LogType = LogType.TRAINING) -> pd.DataFrame:
        """Per-iteration simtrace timing statistics.

        Args:
            log_type:
                ``LogType.TRAINING`` (default) or ``LogType.EVALUATION``.

        Returns:
            A :class:`~pandas.DataFrame` with one row per simtrace file.
            Returns an empty DataFrame when no files are found or none yield
            valid step deltas.

            Columns:

            * **iteration** – Iteration index (``int``, nullable).
            * **stream** – Evaluation run identifier (evaluation only).
            * **file** – Source file path / key.
            * **count** – Number of in-episode step deltas.
            * **avg_ms** – Mean step delta in milliseconds.
            * **max_ms** – Maximum step delta in milliseconds.
            * **p95_ms** – 95th-percentile step delta in milliseconds.
            * **std_ms** – Standard deviation of step deltas in milliseconds.
            * **rtf** – Real-time factor (``float`` or ``None``).
        """
        if log_type == LogType.TRAINING:
            path_attr = self._fh.training_simtrace_path
            split_re = None
        elif log_type == LogType.EVALUATION:
            path_attr = self._fh.evaluation_simtrace_path
            split_attr = self._fh.evaluation_simtrace_split
            split_re = re.compile(split_attr) if split_attr else None
        else:
            raise ValueError(f"Unsupported log_type: {log_type}")

        base_cols = ["iteration", "file", "count", "avg_ms", "max_ms", "p95_ms", "std_ms", "rtf"]
        empty_cols = (
            ["iteration", "stream"] + base_cols[1:] if log_type == LogType.EVALUATION else base_cols
        )

        if path_attr is None:
            return pd.DataFrame(columns=empty_cols)

        files = self._fh.list_files(filterexp=path_attr)
        if not files:
            return pd.DataFrame(columns=empty_cols)

        rows = []

        for file in files:
            try:
                data = self._fh.get_file(file)
                per_episode_deltas, rtf = parse_simtrace_bytes(data)
            except (ValueError, KeyError, csv.Error, OSError) as exc:
                warnings.warn(f"Skipping {file}: {exc}", stacklevel=2)
                continue

            flat = _flatten(per_episode_deltas)
            if flat.size == 0:
                continue

            stats = _summarize(flat)
            row = {
                "iteration": _extract_iteration(file),
                "file": file,
                "count": stats["count"],
                "avg_ms": stats["avg"] * _MS_PER_SECOND,
                "max_ms": stats["max"] * _MS_PER_SECOND,
                "p95_ms": stats["p95"] * _MS_PER_SECOND,
                "std_ms": stats["std"] * _MS_PER_SECOND,
                "rtf": rtf,
            }

            if log_type == LogType.EVALUATION:
                stream = None
                if split_re:
                    m = split_re.search(file)
                    if m and m.lastindex >= 1:
                        stream = m.group(1)
                row["stream"] = stream

            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=empty_cols)

        df = pd.DataFrame(rows)
        df["iteration"] = pd.array(df["iteration"], dtype=pd.Int64Dtype())

        if log_type == LogType.EVALUATION and "stream" in df.columns:
            sort_cols = ["stream", "iteration"]
        else:
            sort_cols = ["iteration"]

        return df.sort_values(sort_cols, kind="stable", na_position="last").reset_index(drop=True)

    def print_summary(self, log_type: LogType = LogType.TRAINING) -> None:
        """Print a human-readable per-iteration stability summary.

        Calls :meth:`analyze` and prints a fixed-width table with one row per
        simtrace file, followed by an aggregate ``OVERALL`` row.

        Args:
            log_type:
                ``LogType.TRAINING`` (default) or ``LogType.EVALUATION``.
        """
        df = self.analyze(log_type)
        if df.empty:
            print("No simtrace files found.")
            return

        header = f"{'label':>12} {'steps':>8} {'avg_ms':>8} {'max_ms':>8} {'p95_ms':>8} {'std_ms':>8} {'rtf':>7}"
        print(header)
        print("-" * len(header))

        for _, row in df.iterrows():
            if log_type == LogType.EVALUATION and "stream" in df.columns:
                stream_val = str(row["stream"]) if pd.notna(row.get("stream")) else "n/a"
                iter_val = str(row["iteration"]) if pd.notna(row.get("iteration")) else "n/a"
                label = f"{stream_val}/{iter_val}"
            else:
                label = str(row["iteration"]) if pd.notna(row["iteration"]) else "n/a"
            rtf = f"{row['rtf']:.3f}" if pd.notna(row.get("rtf")) else "n/a"
            print(
                f"{label:>12} {int(row['count']):>8d} {row['avg_ms']:>8.1f}"
                f" {row['max_ms']:>8.1f} {row['p95_ms']:>8.1f} {row['std_ms']:>8.1f} {rtf:>7}"
            )

        print("-" * len(header))
        total_steps = int(df["count"].sum())
        # weighted average for avg/std; true max; mean of per-file p95 values
        weights = df["count"].values
        wavg = float(np.average(df["avg_ms"].values, weights=weights))
        wstd = float(np.average(df["std_ms"].values, weights=weights))
        overall_max = float(df["max_ms"].max())
        overall_mean_p95 = float(df["p95_ms"].mean())
        rtf_vals = df["rtf"].dropna()
        overall_rtf = f"{rtf_vals.mean():.3f}" if not rtf_vals.empty else "n/a"
        print(
            f"{'OVERALL*':>12} {total_steps:>8d} {wavg:>8.1f}"
            f" {overall_max:>8.1f} {overall_mean_p95:>8.1f} {wstd:>8.1f} {overall_rtf:>7}"
        )
        print(
            "* Note: OVERALL p95_ms is the mean of per-file p95_ms values, not the"
            " global 95th percentile across all step deltas."
        )

    def analyze_episodes(self, file_key: str) -> pd.DataFrame:
        """Per-episode step-delta statistics for a single simtrace file.

        Useful for examining how step timing evolves across episodes within one
        iteration file — e.g. to detect drift or spikes episode-by-episode.

        Args:
            file_key:
                The file path / key as returned by :meth:`analyze` (``file``
                column), or any key accepted by the underlying
                :class:`~deepracer.logs.FileHandler`.

        Returns:
            A :class:`~pandas.DataFrame` with one row per episode.  Columns:
            **episode**, **count**, **avg_ms**, **max_ms**, **p95_ms**,
            **std_ms**.
        """
        return episode_stats(self._fh.get_file(file_key))


def episode_stats(data: bytes) -> pd.DataFrame:
    """Per-episode step-delta statistics from a single simtrace CSV.

    Parses *data* with :func:`parse_simtrace_bytes` and returns one row per
    episode, suitable for trend / evolution analysis across episodes within a
    single iteration file.

    Args:
        data: Raw bytes of a simtrace iteration CSV.

    Returns:
        A :class:`~pandas.DataFrame` with columns **episode**, **count**,
        **avg_ms**, **max_ms**, **p95_ms**, **std_ms**, sorted by episode.

    Raises:
        ValueError: If required columns are missing (see
            :func:`parse_simtrace_bytes`).
    """
    per_episode_deltas, _ = parse_simtrace_bytes(data)
    rows = []
    for ep in sorted(per_episode_deltas):
        deltas = per_episode_deltas[ep]
        if deltas.size == 0:
            continue
        s = _summarize(deltas)
        rows.append(
            {
                "episode": ep,
                "count": s["count"],
                "avg_ms": s["avg"] * _MS_PER_SECOND,
                "max_ms": s["max"] * _MS_PER_SECOND,
                "p95_ms": s["p95"] * _MS_PER_SECOND,
                "std_ms": s["std"] * _MS_PER_SECOND,
            }
        )
    return pd.DataFrame(rows, columns=["episode", "count", "avg_ms", "max_ms", "p95_ms", "std_ms"])
