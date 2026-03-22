import matplotlib

matplotlib.use("Agg")  # non-interactive backend must be set before pyplot is imported

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon as MplPolygon
from unittest import mock

from deepracer.logs import PlottingUtils
from deepracer.tracks import TrackIO

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TRACK_BASE_PATH = "./deepracer/track_utils/tracks"
MODERN_BG_COLOR = "#b0b0b0"
MODERN_ROAD_COLOR = "#484848"


@pytest.fixture(scope="module")
def track():
    """Real reinvent_base track loaded from the test data directory."""
    tu = TrackIO(base_path=TRACK_BASE_PATH)
    return tu.load_track("reinvent_base")


@pytest.fixture(scope="module")
def episode_df():
    """Synthetic DataFrame with three episodes in reinvent_base coordinate range."""
    rows = []
    for ep in range(3):
        t = np.linspace(0, 2 * np.pi, 25)
        for i, angle in enumerate(t):
            rows.append(
                {
                    "episode": ep,
                    "unique_episode": ep,
                    "x": 3.2 + 0.4 * np.cos(angle + ep * 0.2),
                    "y": 0.7 + 0.25 * np.sin(angle + ep * 0.2),
                    "speed": 1.5,
                    "reward": 1.0,
                    "progress": min(100.0, i * 4.0),
                    "tstamp": float(i),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests – classic style
# ---------------------------------------------------------------------------


class TestPlottingUtilsClassic:
    """plot_laps(style='classic') should behave identically to plot_selected_laps()."""

    def test_classic_runs_without_error_with_list(self, track, episode_df):
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps([0, 1], episode_df, track, style="classic")

    def test_classic_runs_without_error_with_dataframe(self, track, episode_df):
        """sorted_idx may be a DataFrame containing the section column."""
        agg = episode_df.groupby("episode", as_index=False).agg(progress=("progress", "max"))
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps(agg, episode_df, track, style="classic")

    def test_classic_delegates_to_plot_selected_laps(self, track, episode_df):
        """plot_laps(style='classic') must call plot_selected_laps internally."""
        with mock.patch.object(
            PlottingUtils, "plot_selected_laps", wraps=PlottingUtils.plot_selected_laps
        ) as spy:
            with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
                PlottingUtils.plot_laps([0], episode_df, track, style="classic")
            spy.assert_called_once()

    def test_classic_is_default_style(self, track, episode_df):
        """Omitting the style parameter should behave identically to style='classic'."""
        with mock.patch.object(
            PlottingUtils, "plot_selected_laps", wraps=PlottingUtils.plot_selected_laps
        ) as spy:
            with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
                PlottingUtils.plot_laps([0], episode_df, track)
            spy.assert_called_once()


# ---------------------------------------------------------------------------
# Tests – modern style
# ---------------------------------------------------------------------------


class TestPlottingUtilsModern:
    """plot_laps(style='modern') should render physically-sized lines."""

    # ------------------------------------------------------------------
    # Smoke / no-error tests
    # ------------------------------------------------------------------

    def test_modern_runs_without_error_with_list(self, track, episode_df):
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps([0, 1], episode_df, track, style="modern")

    def test_modern_runs_without_error_with_dataframe(self, track, episode_df):
        agg = episode_df.groupby("episode", as_index=False).agg(progress=("progress", "max"))
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps(agg, episode_df, track, style="modern")

    def test_modern_single_plot_runs_without_error(self, track, episode_df):
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps([0, 1, 2], episode_df, track, style="modern", single_plot=True)

    def test_modern_accepts_custom_kwargs(self, track, episode_df):
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps(
                [0],
                episode_df,
                track,
                style="modern",
                car_width_m=0.2,
                alpha=0.5,
                color="blue",
                highlight_color="cyan",
            )

    # ------------------------------------------------------------------
    # Visual / structural checks
    # ------------------------------------------------------------------

    def test_modern_figure_background_color(self, track, episode_df):
        """Figure background should be the light-gray off-track color (#b0b0b0)."""
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps([0], episode_df, track, style="modern")
            fig = plt.gcf()
            assert np.allclose(fig.get_facecolor(), to_rgba(MODERN_BG_COLOR), atol=1e-3)

    def test_modern_axes_background_color(self, track, episode_df):
        """Every axes background should be the light-gray off-track color (#b0b0b0)."""
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps([0, 1], episode_df, track, style="modern")
            fig = plt.gcf()
            for ax in fig.get_axes():
                assert np.allclose(ax.get_facecolor(), to_rgba(MODERN_BG_COLOR), atol=1e-3)

    def test_modern_road_patch_present(self, track, episode_df):
        """A filled MplPolygon representing the road surface must be added."""
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps([0], episode_df, track, style="modern")
            ax = plt.gcf().get_axes()[0]
            polygons = [p for p in ax.patches if isinstance(p, MplPolygon)]
            assert len(polygons) >= 1
            road_patch = polygons[0]
            assert np.allclose(road_patch.get_facecolor(), to_rgba(MODERN_ROAD_COLOR), atol=1e-3)

    def test_modern_subplot_count_multi_plot(self, track, episode_df):
        """One subplot per episode when single_plot=False (the default)."""
        episodes = [0, 1, 2]
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps(episodes, episode_df, track, style="modern")
            fig = plt.gcf()
            assert len(fig.get_axes()) == len(episodes)

    def test_modern_subplot_count_single_plot(self, track, episode_df):
        """Exactly one subplot when single_plot=True."""
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps([0, 1, 2], episode_df, track, style="modern", single_plot=True)
            fig = plt.gcf()
            assert len(fig.get_axes()) == 1

    def test_modern_border_and_episode_lines_present(self, track, episode_df):
        """At least borders (inner + outer + centre) plus episode body/spine lines."""
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
            PlottingUtils.plot_laps([0], episode_df, track, style="modern")
            ax = plt.gcf().get_axes()[0]
            lines = ax.get_lines()
            # 3 border/center lines  +  at least 4 per episode (body, spine, start, end)
            assert len(lines) >= 7

    def test_modern_no_classic_delegation(self, track, episode_df):
        """plot_laps(style='modern') must NOT call plot_selected_laps."""
        with mock.patch.object(PlottingUtils, "plot_selected_laps") as no_classic:
            with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.clf"):
                PlottingUtils.plot_laps([0], episode_df, track, style="modern")
            no_classic.assert_not_called()
