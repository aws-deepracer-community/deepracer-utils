"""Simtrace timing stability analysis."""

from __future__ import annotations

import csv
import os
import re
from io import StringIO

import numpy as np
import pandas as pd

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


def _episode_deltas(ep_group: pd.DataFrame) -> np.ndarray:
    """Compute filtered tstamp step deltas for a single episode group.

    Mirrors the filtering in :func:`parse_simtrace_bytes`: every delta is kept
    except the one that *arrives at* a ``prepare`` row (the episode reset gap)
    and any negative deltas (clock resets).
    """
    sorted_group = ep_group.sort_values("steps")
    tstamps = sorted_group["tstamp"].to_numpy(dtype=np.float64)
    statuses = sorted_group["episode_status"].str.lower().str.strip().to_numpy()
    if len(tstamps) < 2:
        return np.array([], dtype=np.float64)
    deltas = np.diff(tstamps)
    # Drop the delta arriving AT a 'prepare' row (episode reset gap).
    keep = np.array([statuses[i + 1] != "prepare" for i in range(len(deltas))])
    deltas = deltas[keep]
    return deltas[deltas >= 0]


def _rtf_from_iteration(it_group: pd.DataFrame) -> float | None:
    """Compute real-time factor for a per-iteration DataFrame group.

    Returns the ratio of simulated time to wall-clock time, or ``None`` when
    no valid ``wall_clock`` data is present.
    """
    mask = it_group["tstamp"].notna() & it_group["wall_clock"].notna()
    filtered = it_group[mask].sort_values("tstamp")
    if len(filtered) < 2:
        return None
    pairs = filtered[["tstamp", "wall_clock"]].to_numpy(dtype=np.float64)
    sim_sum = wall_sum = 0.0
    for i in range(1, len(pairs)):
        ps, pw = pairs[i - 1]
        cs, cw = pairs[i]
        ds, dw = cs - ps, cw - pw
        if ds >= 0 and dw > 0:
            sim_sum += ds
            wall_sum += dw
    return (sim_sum / wall_sum) if wall_sum > 0 else None


def parse_simtrace_bytes(data: bytes) -> tuple:
    """Parse a simtrace CSV byte payload into per-episode step deltas.

    All rows are included in the timestamp sequence regardless of
    ``episode_status``.  The single exception is the delta that *arrives*
    at a ``prepare`` row (the inter-episode reset gap), which is excluded.
    The delta *from* a ``prepare`` row to the next row is kept.

    If the CSV contains a ``wall_clock`` column the real-time factor (RTF)
    is also computed as simulated-seconds / wall-clock-seconds, and the
    first and last wall-clock timestamps are returned.

    Args:
        data:
            Raw bytes of a simtrace iteration CSV with a header row containing
            at least the columns ``episode``, ``tstamp``, and ``episode_status``.

    Returns:
        A 3-tuple ``(per_episode_deltas, real_time_factor, wall_clock_range)``
        where ``per_episode_deltas`` is a ``dict[int, np.ndarray]`` of
        non-negative step deltas **in seconds**, ``real_time_factor`` is a
        ``float`` or ``None`` when wall-clock data is unavailable, and
        ``wall_clock_range`` is a ``(first_wc, last_wc)`` pair of ``float``
        values (or ``(None, None)`` when no wall-clock data is present).

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

    first_wc = sim_wall_points[0][1] if sim_wall_points else None
    last_wc = sim_wall_points[-1][1] if sim_wall_points else None

    return per_episode_deltas, rtf, (first_wc, last_wc)


class SimtraceStabilityAnalyzer:
    """Analyses simtrace timing stability from a loaded trace DataFrame.

    Computes per-iteration, per-worker step-delta statistics from the
    ``tstamp`` column.  All row statuses are included except for the reset gap
    that arrives *at* a ``prepare`` row.

    The DataFrame is produced by
    :meth:`~deepracer.logs.DeepRacerLog.load_training_trace` (training) or
    :meth:`~deepracer.logs.DeepRacerLog.load_evaluation_trace` (evaluation).

    Example::

        from deepracer.logs import DeepRacerLog

        log = DeepRacerLog("./my-model")
        log.load_training_trace()
        df = log.stability.analyze()   # training

    For evaluation::

        log.load_evaluation_trace(ignore_metadata=True)
        eval_analyzer = SimtraceStabilityAnalyzer(log.df)
        df_eval = eval_analyzer.analyze()

    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def analyze(self) -> pd.DataFrame:
        """Per-iteration, per-worker simtrace timing statistics.

        For training data each worker is analysed independently, yielding one
        row per ``(worker, iteration)`` pair.  For evaluation data (identified
        by the presence of a ``stream`` column) each stream/iteration pair
        produces one row.

        Returns:
            A :class:`~pandas.DataFrame` sorted by ``(iteration, worker)`` for
            training or ``(stream, iteration)`` for evaluation.  Returns an
            empty DataFrame when no valid step deltas are found.

            Columns (training):

            * **worker** – Worker index (``int``).
            * **iteration** – Iteration index (``int``, nullable).
            * **count** – Number of in-episode step deltas.
            * **avg_ms** – Mean step delta in milliseconds.
            * **max_ms** – Maximum step delta in milliseconds.
            * **p95_ms** – 95th-percentile step delta in milliseconds.
            * **std_ms** – Standard deviation of step deltas in milliseconds.
            * **rtf** – Real-time factor (``float`` or ``None``).

            Columns (evaluation): ``stream`` replaces ``worker``.
        """
        df = self._df
        has_stream = "stream" in df.columns

        if has_stream:
            group_cols = ["stream", "iteration"]
            base_cols = [
                "stream",
                "iteration",
                "count",
                "avg_ms",
                "max_ms",
                "p95_ms",
                "std_ms",
                "rtf",
            ]
            sort_cols = ["stream", "iteration"]
        else:
            group_cols = ["worker", "iteration"]
            base_cols = [
                "worker",
                "iteration",
                "count",
                "avg_ms",
                "max_ms",
                "p95_ms",
                "std_ms",
                "rtf",
            ]
            sort_cols = ["iteration", "worker"]

        if df.empty:
            return pd.DataFrame(columns=base_cols)

        rows = []
        for (key0, key1), it_group in df.groupby(group_cols, sort=False):
            all_deltas = [
                d
                for _, ep_group in it_group.groupby("episode", sort=False)
                for d in (_episode_deltas(ep_group),)
                if d.size > 0
            ]
            if not all_deltas:
                continue

            flat = np.concatenate(all_deltas)
            stats = _summarize(flat)
            row = {
                "count": stats["count"],
                "avg_ms": stats["avg"] * _MS_PER_SECOND,
                "max_ms": stats["max"] * _MS_PER_SECOND,
                "p95_ms": stats["p95"] * _MS_PER_SECOND,
                "std_ms": stats["std"] * _MS_PER_SECOND,
                "rtf": _rtf_from_iteration(it_group),
            }
            if has_stream:
                row["stream"] = key0
                row["iteration"] = key1
            else:
                row["worker"] = key0
                row["iteration"] = key1
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=base_cols)

        result = pd.DataFrame(rows)
        result["iteration"] = pd.array(result["iteration"], dtype=pd.Int64Dtype())
        if not has_stream:
            result["worker"] = pd.array(result["worker"], dtype=pd.Int64Dtype())
        return result.sort_values(sort_cols, kind="stable", na_position="last").reset_index(
            drop=True
        )[base_cols]

    def print_summary(self) -> None:
        """Print a human-readable per-iteration stability summary.

        Prints a fixed-width table with one row per worker/iteration (or
        stream/iteration for evaluation), followed by an aggregate ``OVERALL``
        row.  When wall-clock data is available a per-iteration timing table
        is appended automatically.
        """
        df = self.analyze()
        if df.empty:
            print("No simtrace data found.")
            return

        has_stream = "stream" in df.columns
        header = (
            f"{'label':>20} {'steps':>8} {'avg_ms':>8} "
            f"{'max_ms':>8} {'p95_ms':>8} {'std_ms':>8} {'rtf':>7}"
        )
        print(header)
        print("-" * len(header))

        for _, row in df.iterrows():
            if has_stream:
                key0 = str(row["stream"]) if pd.notna(row.get("stream")) else "n/a"
            else:
                key0 = str(int(row["worker"])) if pd.notna(row.get("worker")) else "n/a"
            iter_val = str(row["iteration"]) if pd.notna(row.get("iteration")) else "n/a"
            label = f"{key0}/{iter_val}"
            rtf = f"{row['rtf']:.3f}" if pd.notna(row.get("rtf")) else "n/a"
            print(
                f"{label:>20} {int(row['count']):>8d} {row['avg_ms']:>8.1f}"
                f" {row['max_ms']:>8.1f} {row['p95_ms']:>8.1f} {row['std_ms']:>8.1f} {rtf:>7}"
            )

        print("-" * len(header))
        total_steps = int(df["count"].sum())
        # weighted average for avg; true global std via law of total variance;
        # true max; mean of per-row p95 values
        weights = df["count"].values
        wavg = float(np.average(df["avg_ms"].values, weights=weights))
        # Combine per-iteration variances into a true global std.
        # Let μ_i = avg_ms, σ_i = std_ms, n_i = count. Then:
        #   μ = Σ n_i μ_i / N
        #   Var = [Σ n_i (σ_i² + μ_i²) / N] - μ²
        mean_of_squares = np.average(
            df["std_ms"].values ** 2 + df["avg_ms"].values ** 2,
            weights=weights,
        )
        variance = float(mean_of_squares - wavg**2)
        wstd = float(np.sqrt(variance)) if variance > 0.0 else 0.0
        overall_max = float(df["max_ms"].max())
        overall_mean_p95 = float(df["p95_ms"].mean())
        rtf_vals = df["rtf"].dropna()
        overall_rtf = f"{rtf_vals.mean():.3f}" if not rtf_vals.empty else "n/a"
        print(
            f"{'OVERALL':>20} {total_steps:>8d} {wavg:>8.1f}"
            f" {overall_max:>8.1f} {overall_mean_p95:>8.1f} {wstd:>8.1f} {overall_rtf:>7}"
        )

        timing_df = self.analyze_timing()
        if not timing_df.empty:
            print()
            self.print_timing_summary()

    def analyze_timing(self) -> pd.DataFrame:
        """Per-iteration training time and policy update/evaluation time.

        Uses **worker 0** to derive wall-clock timestamps, which represents the
        continuous wall-clock stream across iterations.  Requires a
        ``wall_clock`` column in the trace (provided automatically by
        :meth:`~deepracer.logs.DeepRacerLog.load_training_trace`; set to
        ``NaN`` for older logs that predate it).

        *Training Time* is the elapsed wall-clock time between the first and
        last step of an iteration.  *Policy Update and Evaluation Time* is
        the gap between the last step of iteration *n* and the first step of
        iteration *n+1* — this covers the time spent updating the policy and
        running evaluation.  The *ratio* is Training Time divided by Policy
        Update and Evaluation Time.

        Returns:
            A :class:`~pandas.DataFrame` with one row per iteration.
            Returns an empty DataFrame when no ``wall_clock`` data is
            available.

            Columns:

            * **iteration** – Iteration index (``int``, nullable).
            * **train_time_s** – Training time in seconds (wall-clock).
            * **policy_time_s** – Policy update and evaluation time in
              seconds; ``None`` for the last iteration.
            * **ratio** – ``train_time_s / policy_time_s``; ``None`` when
              ``policy_time_s`` is unavailable or zero.
        """
        df = self._df
        base_cols = ["iteration", "train_time_s", "policy_time_s", "ratio"]

        # Timing analysis is only meaningful for training data; evaluation
        # traces have a ``stream`` column and mix multiple streams per iteration.
        if df.empty or "stream" in df.columns:
            return pd.DataFrame(columns=base_cols)

        # Use worker 0 for the wall-clock timeline across iterations.
        if "worker" in df.columns:
            df = df[df["worker"] == 0]

        rows = []
        for iteration, it_group in df.groupby("iteration", sort=False):
            valid_wc = it_group.dropna(subset=["wall_clock"])
            if valid_wc.empty:
                continue
            rows.append(
                {
                    "iteration": iteration,
                    "_first_wc": float(valid_wc["wall_clock"].min()),
                    "_last_wc": float(valid_wc["wall_clock"].max()),
                }
            )

        if not rows:
            return pd.DataFrame(columns=base_cols)

        result = pd.DataFrame(rows)
        result["iteration"] = pd.array(result["iteration"], dtype=pd.Int64Dtype())
        result = result.sort_values("iteration", kind="stable", na_position="last").reset_index(
            drop=True
        )

        result["train_time_s"] = result["_last_wc"] - result["_first_wc"]

        policy_times: list = [None] * len(result)
        for i in range(len(result) - 1):
            gap = result.iloc[i + 1]["_first_wc"] - result.iloc[i]["_last_wc"]
            # A negative gap can occur due to clock skew; treat as unavailable.
            policy_times[i] = gap if gap >= 0 else None
        result["policy_time_s"] = policy_times

        result["ratio"] = result.apply(
            lambda row: (
                float(row["train_time_s"]) / float(row["policy_time_s"])
                if pd.notna(row["policy_time_s"]) and row["policy_time_s"] != 0
                else None
            ),
            axis=1,
        )

        return result[base_cols]

    def print_timing_summary(self) -> None:
        """Print a human-readable per-iteration timing summary.

        Shows Training Time, Policy Update and Evaluation Time, and their
        ratio for each iteration.  Requires ``wall_clock`` data in the trace.
        Uses worker 0 for the wall-clock timeline.
        """
        df = self.analyze_timing()
        if df.empty:
            print("No timing data available (wall_clock column required).")
            return

        header = f"{'iter':>6} {'train_s':>10} {'policy_s':>10} {'ratio':>8}"
        print(header)
        print("-" * len(header))

        for _, row in df.iterrows():
            iter_val = str(row["iteration"]) if pd.notna(row.get("iteration")) else "n/a"
            train_s = f"{row['train_time_s']:.1f}" if pd.notna(row.get("train_time_s")) else "n/a"
            policy_s = (
                f"{row['policy_time_s']:.1f}" if pd.notna(row.get("policy_time_s")) else "n/a"
            )
            ratio = f"{row['ratio']:.2f}" if pd.notna(row.get("ratio")) else "n/a"
            print(f"{iter_val:>6} {train_s:>10} {policy_s:>10} {ratio:>8}")

        print("-" * len(header))
        avg_train = df["train_time_s"].mean()
        valid_policy = df["policy_time_s"].dropna()
        avg_policy = valid_policy.mean() if not valid_policy.empty else None
        valid_ratio = df["ratio"].dropna()
        avg_ratio = valid_ratio.mean() if not valid_ratio.empty else None

        avg_train_s = f"{avg_train:.1f}" if pd.notna(avg_train) else "n/a"
        avg_policy_s = (
            f"{avg_policy:.1f}" if avg_policy is not None and pd.notna(avg_policy) else "n/a"
        )
        avg_ratio_s = f"{avg_ratio:.2f}" if avg_ratio is not None and pd.notna(avg_ratio) else "n/a"
        print(f"{'AVG':>6} {avg_train_s:>10} {avg_policy_s:>10} {avg_ratio_s:>8}")

    def analyze_episodes(self, iteration: int, worker: int = 0) -> pd.DataFrame:
        """Per-episode step-delta statistics for a single iteration and worker.

        Useful for examining how step timing evolves across episodes within one
        iteration — e.g. to detect drift or spikes episode-by-episode.

        Args:
            iteration:
                The iteration number to analyse.
            worker:
                The worker index to analyse (default: ``0``).  Evaluation
                DataFrames always carry ``worker=0`` so the default is correct
                for both training and evaluation traces.

        Returns:
            A :class:`~pandas.DataFrame` with one row per episode.  Columns:
            **episode**, **count**, **avg_ms**, **max_ms**, **p95_ms**,
            **std_ms**.
        """
        columns = ["episode", "count", "avg_ms", "max_ms", "p95_ms", "std_ms"]
        if "worker" not in self._df.columns:
            raise ValueError(
                "The trace DataFrame has no 'worker' column. "
                "Load the trace via DeepRacerLog.load_training_trace() or "
                "load_evaluation_trace() before creating the analyzer."
            )
        it_df = self._df[(self._df["iteration"] == iteration) & (self._df["worker"] == worker)]
        if it_df.empty:
            return pd.DataFrame(columns=columns)

        rows = []
        for ep, ep_group in it_df.groupby("episode", sort=True):
            deltas = _episode_deltas(ep_group)
            if deltas.size == 0:
                continue
            s = _summarize(deltas)
            rows.append(
                {
                    "episode": int(ep),
                    "count": s["count"],
                    "avg_ms": s["avg"] * _MS_PER_SECOND,
                    "max_ms": s["max"] * _MS_PER_SECOND,
                    "p95_ms": s["p95"] * _MS_PER_SECOND,
                    "std_ms": s["std"] * _MS_PER_SECOND,
                }
            )
        return pd.DataFrame(rows, columns=columns)


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
    per_episode_deltas, _, _ = parse_simtrace_bytes(data)
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
