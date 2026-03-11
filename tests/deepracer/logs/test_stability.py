"""Unit tests for deepracer.logs.stability."""

import os

import numpy as np
import pandas as pd
import pytest

from deepracer.logs import DeepRacerLog, FSFileHandler, TarFileHandler, LogType
from deepracer.logs.stability import (
    SimtraceStabilityAnalyzer,
    _extract_iteration,
    _flatten,
    _rtf_from_iteration,
    _summarize,
    episode_stats,
    parse_simtrace_bytes,
)

# ---------------------------------------------------------------------------
# Helper: build a minimal in-memory simtrace CSV
# ---------------------------------------------------------------------------


def _make_csv(rows, include_wall_clock=False):
    """Return bytes for a minimal simtrace CSV with controlled content.

    *rows* is a list of dicts with keys: episode, tstamp, episode_status,
    and optionally wall_clock.
    """
    fieldnames = [
        "episode",
        "steps",
        "x",
        "y",
        "yaw",
        "steering_angle",
        "speed",
        "action",
        "reward",
        "done",
        "on_track",
        "progress",
        "closest_waypoint",
        "track_len",
        "tstamp",
        "episode_status",
        "pause_duration",
    ]
    if include_wall_clock:
        fieldnames.append("wall_clock")

    lines = [",".join(fieldnames)]
    for i, r in enumerate(rows):
        step = i + 1
        base = f"{r['episode']},{step},0,0,0,0,1,0,1.0,False,True,50,0,60,{r['tstamp']},{r['episode_status']},0.0"
        if include_wall_clock:
            base += f",{r.get('wall_clock', 0)}"
        lines.append(base)
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Tests for parse_simtrace_bytes
# ---------------------------------------------------------------------------


class TestParseSimtraceBytes:
    def test_basic_deltas(self):
        """Known tstamp values produce the expected deltas."""
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 10.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 10.1, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 10.3, "episode_status": "in_progress"},
            ]
        )
        deltas, rtf, _ = parse_simtrace_bytes(data)
        assert 0 in deltas
        np.testing.assert_allclose(deltas[0], [0.1, 0.2], atol=1e-9)
        assert rtf is None

    def test_only_prepare_arrival_excluded(self):
        """Only the delta arriving AT a prepare row is dropped; off_track and pause are kept."""
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 1.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 2.0, "episode_status": "off_track"},  # included
                {"episode": 0, "tstamp": 3.0, "episode_status": "in_progress"},
                {"episode": 1, "tstamp": 5.0, "episode_status": "pause"},  # included
                {
                    "episode": 1,
                    "tstamp": 6.0,
                    "episode_status": "prepare",
                },  # delta arriving here excluded
                {
                    "episode": 1,
                    "tstamp": 7.0,
                    "episode_status": "in_progress",
                },  # delta FROM prepare kept
            ]
        )
        deltas, _, _ = parse_simtrace_bytes(data)
        # Episode 0: all three rows kept → deltas [1.0, 1.0]
        assert 0 in deltas
        np.testing.assert_allclose(deltas[0], [1.0, 1.0], atol=1e-9)
        # Episode 1: delta pause→prepare (1.0) dropped, delta prepare→in_progress (1.0) kept
        assert 1 in deltas
        np.testing.assert_allclose(deltas[1], [1.0], atol=1e-9)

    def test_negative_deltas_excluded(self):
        """Negative deltas (clock resets) are dropped."""
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 10.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 9.5, "episode_status": "in_progress"},  # negative
                {"episode": 0, "tstamp": 11.0, "episode_status": "in_progress"},
            ]
        )
        deltas, _, _ = parse_simtrace_bytes(data)
        assert all(d >= 0 for d in deltas.get(0, np.array([])))

    def test_real_time_factor_computed(self):
        """RTF is computed when wall_clock column is present."""
        data = _make_csv(
            [
                {
                    "episode": 0,
                    "tstamp": 0.0,
                    "episode_status": "in_progress",
                    "wall_clock": 1000.0,
                },
                {
                    "episode": 0,
                    "tstamp": 1.0,
                    "episode_status": "in_progress",
                    "wall_clock": 1001.0,
                },
                {
                    "episode": 0,
                    "tstamp": 2.0,
                    "episode_status": "in_progress",
                    "wall_clock": 1002.0,
                },
            ],
            include_wall_clock=True,
        )
        _, rtf, _ = parse_simtrace_bytes(data)
        assert rtf is not None
        assert pytest.approx(1.0, rel=1e-3) == rtf

    def test_rtf_none_without_wall_clock(self):
        """RTF is None when the wall_clock column is absent."""
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 0.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 1.0, "episode_status": "in_progress"},
            ]
        )
        _, rtf, _ = parse_simtrace_bytes(data)
        assert rtf is None

    def test_missing_required_columns_raises(self):
        csv_text = b"episode,steps\n0,1\n"
        with pytest.raises(ValueError, match="Missing required column"):
            parse_simtrace_bytes(csv_text)

    def test_single_row_episode_has_no_entry(self):
        """An episode with only one in-progress row produces no deltas."""
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 5.0, "episode_status": "in_progress"},
            ]
        )
        deltas, _, _ = parse_simtrace_bytes(data)
        assert 0 not in deltas

    def test_multiple_episodes(self):
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 1.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 1.1, "episode_status": "in_progress"},
                {"episode": 1, "tstamp": 5.0, "episode_status": "in_progress"},
                {"episode": 1, "tstamp": 5.2, "episode_status": "in_progress"},
                {"episode": 1, "tstamp": 5.5, "episode_status": "in_progress"},
            ]
        )
        deltas, _, _ = parse_simtrace_bytes(data)
        assert set(deltas.keys()) == {0, 1}
        assert deltas[0].size == 1
        assert deltas[1].size == 2


# ---------------------------------------------------------------------------
# Tests for helper utilities
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_iteration_from_named_file(self):
        assert _extract_iteration("0-iteration.csv") == 0
        assert _extract_iteration("12-iteration.csv") == 12
        assert _extract_iteration("/some/path/7-iteration.csv") == 7

    def test_extract_iteration_returns_none_for_unknown(self):
        assert _extract_iteration("some-other-file.csv") is None

    def test_flatten_combines_arrays(self):
        d = {0: np.array([0.1, 0.2]), 1: np.array([0.3])}
        flat = _flatten(d)
        assert flat.size == 3

    def test_flatten_empty_dict(self):
        assert _flatten({}).size == 0

    def test_summarize_basic(self):
        values = np.array([0.1, 0.2, 0.3])
        s = _summarize(values)
        assert s["count"] == 3
        assert pytest.approx(0.2, rel=1e-6) == s["avg"]
        assert s["max"] == pytest.approx(0.3, rel=1e-6)

    def test_summarize_empty_returns_none(self):
        assert _summarize(np.array([])) is None


class TestRtfFromIteration:
    """Unit tests for the _rtf_from_iteration helper."""

    def _make_group(self, tstamps, wall_clocks):
        return pd.DataFrame({"tstamp": tstamps, "wall_clock": wall_clocks})

    def test_basic_rtf(self):
        """RTF = sim_delta / wall_delta over two rows."""
        group = self._make_group([0.0, 0.2], [1000.0, 1002.0])
        # RTF = 0.2 / 2.0 = 0.1
        assert _rtf_from_iteration(group) == pytest.approx(0.1, rel=1e-6)

    def test_out_of_order_rows_are_sorted(self):
        """Rows are sorted by tstamp before computing RTF."""
        group = self._make_group([0.2, 0.0, 0.1], [1002.0, 1000.0, 1001.0])
        # Sorted: (0.0,1000), (0.1,1001), (0.2,1002) → sim 0.2, wall 2.0 → 0.1
        assert _rtf_from_iteration(group) == pytest.approx(0.1, rel=1e-6)

    def test_nan_wall_clock_rows_excluded(self):
        """Rows with NaN wall_clock are excluded."""
        group = self._make_group([0.0, 0.1, 0.2], [1000.0, float("nan"), 1002.0])
        # Only (0.0,1000) and (0.2,1002) contribute → sim 0.2, wall 2.0 → 0.1
        assert _rtf_from_iteration(group) == pytest.approx(0.1, rel=1e-6)

    def test_nan_tstamp_rows_excluded(self):
        """Rows with NaN tstamp are excluded."""
        group = self._make_group([0.0, float("nan"), 0.2], [1000.0, 1001.0, 1002.0])
        assert _rtf_from_iteration(group) == pytest.approx(0.1, rel=1e-6)

    def test_returns_none_when_fewer_than_two_valid_rows(self):
        """Returns None when fewer than two rows have both tstamp and wall_clock."""
        group = self._make_group([0.0], [1000.0])
        assert _rtf_from_iteration(group) is None

    def test_returns_none_when_all_wall_clock_nan(self):
        """Returns None when no valid wall_clock values are present."""
        group = self._make_group([0.0, 0.1, 0.2], [float("nan")] * 3)
        assert _rtf_from_iteration(group) is None


# Integration tests: SimtraceStabilityAnalyzer with FSFileHandler
# ---------------------------------------------------------------------------

BASE = os.path.dirname(__file__)  # absolute path to this file's directory


class TestAnalyzeDrfc1Training:
    @pytest.fixture
    def analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        log.load_training_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    def test_returns_dataframe(self, analyzer):
        df = analyzer.analyze()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, analyzer):
        df = analyzer.analyze()
        for col in ("worker", "iteration", "count", "avg_ms", "max_ms", "p95_ms", "std_ms", "rtf"):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_matches_iteration_files(self, analyzer):
        # sample-drfc-1-logs has 27 training-simtrace CSVs (0..26)
        df = analyzer.analyze()
        assert len(df) == 27

    def test_sorted_by_iteration(self, analyzer):
        df = analyzer.analyze()
        iterations = df["iteration"].dropna().tolist()
        assert iterations == sorted(iterations)

    def test_avg_ms_positive(self, analyzer):
        df = analyzer.analyze()
        assert (df["avg_ms"] > 0).all()

    def test_max_gte_avg(self, analyzer):
        df = analyzer.analyze()
        assert (df["max_ms"] >= df["avg_ms"]).all()

    def test_p95_between_avg_and_max(self, analyzer):
        df = analyzer.analyze()
        assert (df["p95_ms"] >= df["avg_ms"]).all()
        assert (df["max_ms"] >= df["p95_ms"]).all()

    def test_no_rtf_without_wall_clock(self, analyzer):
        df = analyzer.analyze()
        # sample-drfc-1-logs has no wall_clock column
        assert df["rtf"].isna().all()

    def test_snapshot_iteration_0(self, analyzer):
        row = analyzer.analyze().iloc[0]
        assert row["worker"] == 0
        assert row["iteration"] == 0
        assert row["count"] == 509
        assert pytest.approx(66.633, rel=1e-3) == row["avg_ms"]
        assert pytest.approx(93.0, rel=1e-3) == row["max_ms"]
        assert pytest.approx(82.0, rel=1e-3) == row["p95_ms"]
        assert pytest.approx(9.187, rel=1e-3) == row["std_ms"]

    def test_snapshot_iteration_1(self, analyzer):
        row = analyzer.analyze().iloc[1]
        assert row["worker"] == 0
        assert row["iteration"] == 1
        assert row["count"] == 525
        assert pytest.approx(66.438, rel=1e-3) == row["avg_ms"]
        assert pytest.approx(96.0, rel=1e-3) == row["max_ms"]
        assert pytest.approx(82.0, rel=1e-3) == row["p95_ms"]

    def test_snapshot_iteration_26(self, analyzer):
        row = analyzer.analyze().iloc[26]
        assert row["worker"] == 0
        assert row["iteration"] == 26
        assert row["count"] == 2908
        assert pytest.approx(66.879, rel=1e-3) == row["avg_ms"]
        assert pytest.approx(422.0, rel=1e-3) == row["max_ms"]
        assert pytest.approx(83.0, rel=1e-3) == row["p95_ms"]
        assert pytest.approx(13.153, rel=1e-3) == row["std_ms"]


class TestAnalyzeDrfc1Evaluation:
    @pytest.fixture
    def analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        log.load_evaluation_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    def test_evaluation_returns_dataframe(self, analyzer):
        df = analyzer.analyze()
        assert isinstance(df, pd.DataFrame)

    def test_evaluation_has_stream_column(self, analyzer):
        df = analyzer.analyze()
        assert "stream" in df.columns

    def test_evaluation_non_empty(self, analyzer):
        df = analyzer.analyze()
        assert len(df) > 0

    def test_snapshot_evaluation_streams(self, analyzer):
        df = analyzer.analyze()
        assert list(df["stream"]) == ["20220709200242", "20220709200509", "20220709200711"]
        assert list(df["count"]) == [647, 631, 624]
        np.testing.assert_allclose(
            df["avg_ms"].values,
            [66.683, 66.655, 66.663],
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            df["p95_ms"].values,
            [80.0, 79.0, 78.0],
            rtol=1e-3,
        )


class TestAnalyzeDroaTraining:
    @pytest.fixture
    def analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-droa-solution-logs")
        log.load_training_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    def test_returns_dataframe(self, analyzer):
        df = analyzer.analyze()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_rtf_present_for_wall_clock_data(self, analyzer):
        df = analyzer.analyze()
        # sample-droa-solution-logs has wall_clock column
        assert df["rtf"].notna().any()

    def test_snapshot_droa_training(self, analyzer):
        df = analyzer.analyze()
        assert len(df) == 2
        row0 = df.iloc[0]
        assert row0["worker"] == 0
        assert row0["iteration"] == 0
        assert row0["count"] == 353
        assert pytest.approx(65.887, rel=1e-3) == row0["avg_ms"]
        assert pytest.approx(92.0, rel=1e-3) == row0["max_ms"]
        assert pytest.approx(79.0, rel=1e-3) == row0["p95_ms"]
        assert pytest.approx(8.320, rel=1e-3) == row0["std_ms"]
        assert pytest.approx(0.6439, rel=1e-3) == row0["rtf"]
        row1 = df.iloc[1]
        assert row1["worker"] == 0
        assert row1["iteration"] == 1
        assert row1["count"] == 453
        assert pytest.approx(66.322, rel=1e-3) == row1["avg_ms"]
        assert pytest.approx(144.0, rel=1e-3) == row1["max_ms"]
        assert pytest.approx(0.6521, rel=1e-3) == row1["rtf"]


class TestAnalyzeEvalOnly:
    """Evaluation-only folders have no training simtrace."""

    def test_stability_raises_when_not_loaded(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-droa-eval-only")
        with pytest.raises(RuntimeError, match="No trace loaded"):
            _ = log.stability

    def test_evaluation_non_empty(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-droa-eval-only")
        log.load_evaluation_trace(ignore_metadata=True)
        df = SimtraceStabilityAnalyzer(log.df).analyze()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_snapshot_eval_only_evaluation(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-droa-eval-only")
        log.load_evaluation_trace(ignore_metadata=True)
        df = SimtraceStabilityAnalyzer(log.df).analyze()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["count"] == 1299
        assert pytest.approx(67.424, rel=1e-3) == row["avg_ms"]
        assert pytest.approx(150.0, rel=1e-3) == row["max_ms"]
        assert pytest.approx(81.0, rel=1e-3) == row["p95_ms"]
        assert pytest.approx(12.216, rel=1e-3) == row["std_ms"]
        assert pytest.approx(0.7142, rel=1e-3) == row["rtf"]


class TestDeepRacerLogStabilityProperty:
    def test_stability_requires_loaded_trace(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        with pytest.raises(RuntimeError, match="No trace loaded"):
            _ = log.stability

    def test_stability_property_returns_analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        log.load_training_trace(ignore_metadata=True)
        assert isinstance(log.stability, SimtraceStabilityAnalyzer)

    def test_stability_analyze_via_log(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        log.load_training_trace(ignore_metadata=True)
        df = log.stability.analyze()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_stability_evaluate_via_log(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        log.load_evaluation_trace(ignore_metadata=True)
        df = log.stability.analyze()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Integration tests: SimtraceStabilityAnalyzer with TarFileHandler
# ---------------------------------------------------------------------------

SAMPLE_TAR = f"{BASE}/sample-droa-solution-logs.tar.gz"
EVAL_ONLY_TAR = f"{BASE}/sample-droa-eval-only.tar.gz"


class TestAnalyzeTarTraining:
    @pytest.fixture
    def analyzer(self):
        log = DeepRacerLog(filehandler=TarFileHandler(SAMPLE_TAR))
        log.load_training_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    def test_returns_dataframe(self, analyzer):
        df = analyzer.analyze()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, analyzer):
        df = analyzer.analyze()
        for col in ("worker", "iteration", "count", "avg_ms", "max_ms", "p95_ms", "std_ms", "rtf"):
            assert col in df.columns, f"Missing column: {col}"

    def test_non_empty(self, analyzer):
        df = analyzer.analyze()
        # sample-droa-solution-logs.tar.gz has 2 training-simtrace CSVs
        assert len(df) == 2

    def test_sorted_by_iteration(self, analyzer):
        df = analyzer.analyze()
        iterations = df["iteration"].dropna().tolist()
        assert iterations == sorted(iterations)

    def test_avg_ms_positive(self, analyzer):
        df = analyzer.analyze()
        assert (df["avg_ms"] > 0).all()

    def test_rtf_present_for_wall_clock_data(self, analyzer):
        """The DROA tar archive includes wall_clock; RTF should be populated."""
        df = analyzer.analyze()
        assert df["rtf"].notna().any()

    def test_snapshot_tar_training(self, analyzer):
        df = analyzer.analyze()
        row0 = df.iloc[0]
        assert row0["worker"] == 0
        assert row0["iteration"] == 0
        assert row0["count"] == 353
        assert pytest.approx(65.887, rel=1e-3) == row0["avg_ms"]
        assert pytest.approx(92.0, rel=1e-3) == row0["max_ms"]
        assert pytest.approx(79.0, rel=1e-3) == row0["p95_ms"]
        assert pytest.approx(8.320, rel=1e-3) == row0["std_ms"]
        assert pytest.approx(0.6439, rel=1e-3) == row0["rtf"]

    def test_results_match_fs_handler(self):
        """TarFileHandler and FSFileHandler must produce the same stats."""
        log_tar = DeepRacerLog(filehandler=TarFileHandler(SAMPLE_TAR))
        log_tar.load_training_trace(ignore_metadata=True)
        df_tar = SimtraceStabilityAnalyzer(log_tar.df).analyze()

        log_fs = DeepRacerLog(model_folder=f"{BASE}/sample-droa-solution-logs")
        log_fs.load_training_trace(ignore_metadata=True)
        df_fs = SimtraceStabilityAnalyzer(log_fs.df).analyze()

        assert len(df_tar) == len(df_fs)
        for col in ("count", "avg_ms", "max_ms", "p95_ms", "std_ms"):
            np.testing.assert_allclose(
                df_tar[col].values,
                df_fs[col].values,
                rtol=1e-5,
                err_msg=f"Column '{col}' differs between Tar and FS handlers",
            )


class TestAnalyzeTarEvaluation:
    @pytest.fixture
    def analyzer(self):
        log = DeepRacerLog(filehandler=TarFileHandler(SAMPLE_TAR))
        log.load_evaluation_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    def test_evaluation_returns_dataframe(self, analyzer):
        df = analyzer.analyze()
        assert isinstance(df, pd.DataFrame)

    def test_evaluation_has_stream_column(self, analyzer):
        df = analyzer.analyze()
        assert "stream" in df.columns

    def test_evaluation_non_empty(self, analyzer):
        df = analyzer.analyze()
        assert len(df) > 0

    def test_snapshot_tar_evaluation(self, analyzer):
        df = analyzer.analyze()
        assert len(df) == 1
        row = df.iloc[0]
        assert "stream" in df.columns
        assert row["count"] == 1299
        assert pytest.approx(67.424, rel=1e-3) == row["avg_ms"]
        assert pytest.approx(150.0, rel=1e-3) == row["max_ms"]
        assert pytest.approx(81.0, rel=1e-3) == row["p95_ms"]
        assert pytest.approx(12.216, rel=1e-3) == row["std_ms"]
        assert pytest.approx(0.7142, rel=1e-3) == row["rtf"]


class TestAnalyzeTarEvalOnly:
    """Evaluation-only tar archive: stability raises until trace is loaded."""

    def test_stability_raises_when_not_loaded(self):
        log = DeepRacerLog(filehandler=TarFileHandler(EVAL_ONLY_TAR))
        with pytest.raises(RuntimeError, match="No trace loaded"):
            _ = log.stability

    def test_evaluation_non_empty(self):
        log = DeepRacerLog(filehandler=TarFileHandler(EVAL_ONLY_TAR))
        log.load_evaluation_trace(ignore_metadata=True)
        df = SimtraceStabilityAnalyzer(log.df).analyze()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestDeepRacerLogStabilityPropertyTar:
    def test_stability_analyze_via_log_tar(self):
        log = DeepRacerLog(filehandler=TarFileHandler(SAMPLE_TAR))
        log.load_training_trace(ignore_metadata=True)
        df = log.stability.analyze()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_stability_evaluate_via_log_tar(self):
        log = DeepRacerLog(filehandler=TarFileHandler(SAMPLE_TAR))
        log.load_evaluation_trace(ignore_metadata=True)
        df = log.stability.analyze()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Tests for episode_stats() and analyze_episodes()
# ---------------------------------------------------------------------------


class TestEpisodeStats:
    def test_returns_dataframe(self):
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 1.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 1.1, "episode_status": "in_progress"},
                {"episode": 1, "tstamp": 5.0, "episode_status": "in_progress"},
                {"episode": 1, "tstamp": 5.2, "episode_status": "in_progress"},
            ]
        )
        df = episode_stats(data)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 1.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 1.1, "episode_status": "in_progress"},
            ]
        )
        df = episode_stats(data)
        for col in ("episode", "count", "avg_ms", "max_ms", "p95_ms", "std_ms"):
            assert col in df.columns

    def test_one_row_per_episode(self):
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 1.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 1.1, "episode_status": "in_progress"},
                {"episode": 1, "tstamp": 5.0, "episode_status": "in_progress"},
                {"episode": 1, "tstamp": 5.2, "episode_status": "in_progress"},
                {"episode": 2, "tstamp": 9.0, "episode_status": "in_progress"},
                {"episode": 2, "tstamp": 9.3, "episode_status": "in_progress"},
            ]
        )
        df = episode_stats(data)
        assert len(df) == 3
        assert list(df["episode"]) == [0, 1, 2]

    def test_values_in_milliseconds(self):
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 0.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 0.1, "episode_status": "in_progress"},  # 100 ms
            ]
        )
        df = episode_stats(data)
        assert pytest.approx(100.0, rel=1e-6) == df.iloc[0]["avg_ms"]

    def test_empty_for_no_valid_episodes(self):
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 5.0, "episode_status": "in_progress"},  # single row
            ]
        )
        df = episode_stats(data)
        assert df.empty

    def test_integration_real_file(self):
        """episode_stats on a real sample file returns sane per-episode data."""
        with open(
            f"{BASE}/sample-droa-solution-logs/sim-trace/training/"
            "2026-03-06T18:33:59.511Z-deepracerindy-training-ACGVRmRuFNU9NkQ/"
            "training-simtrace/0-iteration.csv",
            "rb",
        ) as f:
            data = f.read()
        df = episode_stats(data)
        assert len(df) > 1
        assert (df["avg_ms"] > 0).all()
        assert (df["max_ms"] >= df["avg_ms"]).all()


_DROA_ITER0_PATH = (
    f"{BASE}/sample-droa-solution-logs/sim-trace/training/"
    "2026-03-06T18:33:59.511Z-deepracerindy-training-ACGVRmRuFNU9NkQ/"
    "training-simtrace/0-iteration.csv"
)


class TestAnalyzeEpisodes:
    @pytest.fixture
    def analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-droa-solution-logs")
        log.load_training_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    def test_returns_dataframe(self, analyzer):
        df = analyzer.analyze_episodes(0)
        assert isinstance(df, pd.DataFrame)

    def test_matches_episode_stats_function(self, analyzer):
        df_method = analyzer.analyze_episodes(0)
        with open(_DROA_ITER0_PATH, "rb") as f:
            df_func = episode_stats(f.read())
        pd.testing.assert_frame_equal(df_method, df_func)


class TestPrintSummary:
    @pytest.fixture
    def analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        log.load_training_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    def test_training_prints_without_error(self, analyzer, capsys):
        analyzer.print_summary()
        out = capsys.readouterr().out
        assert "OVERALL" in out
        assert "avg_ms" in out

    def test_evaluation_prints_without_error(self, capsys):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        log.load_evaluation_trace(ignore_metadata=True)
        SimtraceStabilityAnalyzer(log.df).print_summary()
        out = capsys.readouterr().out
        assert "OVERALL" in out
        assert "20220709200242" in out

    def test_empty_prints_no_data_message(self, capsys):
        SimtraceStabilityAnalyzer(pd.DataFrame()).print_summary()
        out = capsys.readouterr().out
        assert "No simtrace data found" in out


# ---------------------------------------------------------------------------
# Tests for wall_clock_range returned by parse_simtrace_bytes
# ---------------------------------------------------------------------------


class TestParseSimtraceBytesWallClockRange:
    def test_returns_none_without_wall_clock_column(self):
        data = _make_csv(
            [
                {"episode": 0, "tstamp": 10.0, "episode_status": "in_progress"},
                {"episode": 0, "tstamp": 10.1, "episode_status": "in_progress"},
            ]
        )
        _, _, (first, last) = parse_simtrace_bytes(data)
        assert first is None
        assert last is None

    def test_returns_first_and_last(self):
        data = _make_csv(
            [
                {
                    "episode": 0,
                    "tstamp": 10.0,
                    "episode_status": "in_progress",
                    "wall_clock": 1000.0,
                },
                {
                    "episode": 0,
                    "tstamp": 10.1,
                    "episode_status": "in_progress",
                    "wall_clock": 1001.5,
                },
                {
                    "episode": 0,
                    "tstamp": 10.2,
                    "episode_status": "in_progress",
                    "wall_clock": 1003.0,
                },
            ],
            include_wall_clock=True,
        )
        _, _, (first, last) = parse_simtrace_bytes(data)
        assert pytest.approx(1000.0) == first
        assert pytest.approx(1003.0) == last

    def test_single_row_returns_same_for_first_and_last(self):
        data = _make_csv(
            [{"episode": 0, "tstamp": 5.0, "episode_status": "in_progress", "wall_clock": 500.0}],
            include_wall_clock=True,
        )
        _, _, (first, last) = parse_simtrace_bytes(data)
        assert pytest.approx(500.0) == first
        assert pytest.approx(500.0) == last

    def test_integration_real_file(self):
        with open(
            f"{BASE}/sample-droa-solution-logs/sim-trace/training/"
            "2026-03-06T18:33:59.511Z-deepracerindy-training-ACGVRmRuFNU9NkQ/"
            "training-simtrace/0-iteration.csv",
            "rb",
        ) as f:
            data = f.read()
        _, _, (first, last) = parse_simtrace_bytes(data)
        assert first is not None
        assert last is not None
        assert last > first
        assert pytest.approx(1772822267.7927284, rel=1e-6) == first
        assert pytest.approx(1772822309.2516327, rel=1e-6) == last


# ---------------------------------------------------------------------------
# Tests for SimtraceStabilityAnalyzer.analyze_timing
# ---------------------------------------------------------------------------


class TestAnalyzeTiming:
    @pytest.fixture
    def droa_analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-droa-solution-logs")
        log.load_training_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    @pytest.fixture
    def drfc1_analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        log.load_training_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    def test_returns_dataframe(self, droa_analyzer):
        df = droa_analyzer.analyze_timing()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, droa_analyzer):
        df = droa_analyzer.analyze_timing()
        for col in ("iteration", "train_time_s", "policy_time_s", "ratio"):
            assert col in df.columns

    def test_empty_when_no_wall_clock(self, drfc1_analyzer):
        df = drfc1_analyzer.analyze_timing()
        assert df.empty
        for col in ("iteration", "train_time_s", "policy_time_s", "ratio"):
            assert col in df.columns

    def test_sorted_by_iteration(self, droa_analyzer):
        df = droa_analyzer.analyze_timing()
        assert list(df["iteration"]) == sorted(df["iteration"])

    def test_train_time_positive(self, droa_analyzer):
        df = droa_analyzer.analyze_timing()
        assert (df["train_time_s"] > 0).all()

    def test_policy_time_none_for_last_iteration(self, droa_analyzer):
        df = droa_analyzer.analyze_timing()
        assert pd.isna(df.iloc[-1]["policy_time_s"])

    def test_ratio_none_for_last_iteration(self, droa_analyzer):
        df = droa_analyzer.analyze_timing()
        assert pd.isna(df.iloc[-1]["ratio"])

    def test_snapshot_droa_training(self, droa_analyzer):
        df = droa_analyzer.analyze_timing()
        assert len(df) == 2
        row0 = df.iloc[0]
        assert row0["iteration"] == 0
        assert pytest.approx(41.459, rel=1e-3) == row0["train_time_s"]
        assert pytest.approx(13.340, rel=1e-3) == row0["policy_time_s"]
        assert pytest.approx(3.108, rel=1e-3) == row0["ratio"]
        row1 = df.iloc[1]
        assert row1["iteration"] == 1
        assert pytest.approx(51.294, rel=1e-3) == row1["train_time_s"]
        assert pd.isna(row1["policy_time_s"])
        assert pd.isna(row1["ratio"])

    def test_empty_df_returns_empty_timing(self):
        df = SimtraceStabilityAnalyzer(pd.DataFrame()).analyze_timing()
        assert df.empty
        for col in ("iteration", "train_time_s", "policy_time_s", "ratio"):
            assert col in df.columns


# ---------------------------------------------------------------------------
# Tests for SimtraceStabilityAnalyzer.print_timing_summary
# ---------------------------------------------------------------------------


class TestPrintTimingSummary:
    @pytest.fixture
    def droa_analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-droa-solution-logs")
        log.load_training_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    @pytest.fixture
    def drfc1_analyzer(self):
        log = DeepRacerLog(model_folder=f"{BASE}/sample-drfc-1-logs")
        log.load_training_trace(ignore_metadata=True)
        return SimtraceStabilityAnalyzer(log.df)

    def test_prints_without_error(self, droa_analyzer, capsys):
        droa_analyzer.print_timing_summary()
        out = capsys.readouterr().out
        assert "AVG" in out
        assert "train_s" in out

    def test_includes_iter_and_ratio(self, droa_analyzer, capsys):
        droa_analyzer.print_timing_summary()
        out = capsys.readouterr().out
        assert "iter" in out
        assert "ratio" in out

    def test_no_wall_clock_prints_message(self, drfc1_analyzer, capsys):
        drfc1_analyzer.print_timing_summary()
        out = capsys.readouterr().out
        assert "No timing data available" in out

    def test_print_summary_includes_timing_when_wall_clock_available(self, droa_analyzer, capsys):
        droa_analyzer.print_summary()
        out = capsys.readouterr().out
        # Both stability section and timing section should be present
        assert "OVERALL" in out
        assert "AVG" in out
        assert "train_s" in out

    def test_print_summary_no_timing_when_no_wall_clock(self, drfc1_analyzer, capsys):
        drfc1_analyzer.print_summary()
        out = capsys.readouterr().out
        assert "OVERALL" in out
        assert "train_s" not in out
