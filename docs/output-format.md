# Output Data Frame Format

After calling `log.load()`, `log.load_training_trace()`, `log.load_evaluation_trace()`, or `log.load_robomaker_logs()`, the parsed data is available as a [pandas](https://pandas.pydata.org/) `DataFrame` at `log.df`.

This document describes every column, its type, and the valid values.

---

## Training DataFrame

Produced by `load_training_trace()` (simtrace CSV path) and `load_robomaker_logs()` (CONSOLE_MODEL_WITH_LOGS only). Both methods emit the same schema.

Rows are sorted by `unique_episode` then `steps`.

### Column reference

| Column | dtype | Description |
|---|---|---|
| `episode` | `int` | Episode number within the worker and iteration. Resets to 0 at the start of each iteration for each worker. |
| `steps` | `float` | Step counter within the episode. Starts at 1. |
| `x` | `float` | Car centre X coordinate in metres (track coordinate system). |
| `y` | `float` | Car centre Y coordinate in metres (track coordinate system). |
| `yaw` | `float` | Car heading angle in degrees. |
| `steering_angle` | `float` | Commanded steering angle in degrees. Positive is left, negative is right. Derived from the action space. |
| `speed` | `float` | Commanded speed (throttle) in m/s. Derived from the action space. |
| `action` | `int` | Zero-based index of the action selected from the action space. `-1` when the action index is not available (e.g. continuous action spaces or older logs). |
| `reward` | `float` | Reward returned by the reward function at this step. |
| `done` | `bool` | `True` when the episode ended at this step (either success or failure). |
| `on_track` | `bool` | `True` when all four wheels are on the track surface. |
| `progress` | `float` | Percentage of the track completed in this episode, in the range `[0, 100]`. |
| `closest_waypoint` | `int` | Index of the nearest waypoint to the car at this step. |
| `track_len` | `float` | Total track length in metres. Constant within a training run. |
| `tstamp` | `float` | Simulation time as a Unix timestamp (seconds since epoch). |
| `episode_status` | `str` | Status of the episode at this step. See [Episode status values](#episode-status-values). |
| `pause_duration` | `float` | Cumulative pause duration in seconds incurred up to this step. `NaN` for logs produced before this field was introduced. |
| `wall_clock` | `float` | Wall-clock time as a Unix timestamp. `NaN` for logs produced before this field was introduced. |
| `iteration` | `int` | Iteration number, taken from the CSV filename (`{N}-iteration.csv`). Counts from `0`. |
| `worker` | `int` | Worker index. Always `0` for single-worker formats. In DRFC multi-worker format, reflects the numbered subdirectory (`0`, `1`, `2`, …). |
| `unique_episode` | `int` | Globally unique episode index, computed across all workers and iterations so that every episode has a distinct value. Used as the primary sort key. |

### Episode status values

| Value | Meaning |
|---|---|
| `prepare` | The episode is initialising (car is not yet moving). |
| `in_progress` | The car is actively driving. |
| `off_track` | The car has left the track boundary. |
| `lap_complete` | The car has completed a full lap successfully. |
| `crashed` | The car collided with an obstacle. |
| `pause` | The car is paused — either during a race pause event, or after a crash where the car has been reset to the track and is waiting for the penalty timer to expire before resuming. |
| `reversed` | The car is driving in the wrong direction. |
| `time_up` | The maximum episode time was reached. |

> **Note:** Not all statuses appear in every dataset. The final row of an episode typically carries a terminal status (`off_track`, `lap_complete`, `crashed`, etc.) while all preceding rows show `in_progress` or `prepare`.

---

## Evaluation DataFrame

Produced by `load_evaluation_trace()`. This contains all evaluation runs concatenated into a single DataFrame.

Rows are sorted by `stream`, then `episode`, then `steps`.

The evaluation DataFrame has **all the same columns as the training DataFrame**, with the following differences:

| Column | Change compared to training |
|---|---|
| `stream` | **Added.** String identifier for the evaluation run (timestamp or UUID derived from the folder name). Groups rows belonging to the same evaluation. |
| `unique_episode` | **Absent.** Not computed for evaluations. |
| `worker` | Always `0` (evaluations are always single-worker). |

---

## Metrics DataFrame (TrainingMetrics / EvaluationMetrics)

`TrainingMetrics` exposes a separate DataFrame at `metrics.metrics`. It is loaded from the `TrainingMetrics.json` / `EvaluationMetrics-*.json` files and has a different schema from the simtrace DataFrame.

| Column | dtype | Description |
|---|---|---|
| `episode` | `int` | Global episode number within the training run. |
| `trial` | `int` | Episode number within the current iteration (1-based). |
| `phase` | `str` | Training phase: `training` or `evaluation`. |
| `completion_percentage` | `float` | Track completion percentage for the episode, `[0, 100]`. |
| `elapsed_time_in_milliseconds` | `float` | Episode wall-clock duration in milliseconds. |
| `reward_score` | `float` | Total cumulative reward for the episode. |
| `iteration` | `int` | Iteration index (0-based), derived from `episode` and `episodes_per_iteration`. |
| `master_iteration` | `int` | Iteration index adjusted across multiple training rounds loaded via `append_file()`. |
| `round` | `int` | Training round number (1-based), set when appending multiple metric files. |
| `r-i` | `str` | Human-readable composite index (`{round}-{iteration}`) for plot labels; zero-padded. |
| `r-e` | `str` | Human-readable composite episode index (`{round}-{episode}`); zero-padded. |

---

## Column name mapping (CSV → DataFrame)

The raw simtrace CSV files use different column names from those exposed by the library. The table below documents the renaming applied on load:

| CSV column | DataFrame column | Notes |
|---|---|---|
| `X` | `x` | |
| `Y` | `y` | |
| `steer` | `steering_angle` | |
| `throttle` | `speed` | |
| `all_wheels_on_track` | `on_track` | |
| `obstacle_crash_counter` | *(dropped)* | Present in some log formats; removed on load. |

All other CSV column names are carried through unchanged.

---

## Quick reference

```python
from deepracer.logs import DeepRacerLog

log = DeepRacerLog("path/to/model-folder")
log.load()           # auto-selects training or evaluation based on folder type

df = log.df

# Basic exploration
print(df.dtypes)
print(df["episode_status"].value_counts())

# All steps where the car went off-track
off_track = df[df["episode_status"] == "off_track"]

# Progress per unique episode
per_episode = df.groupby("unique_episode")["progress"].max()

# Filter a single evaluation stream
eval_run = df[df["stream"] == "2026-03-07T17:52:00.898Z-deepracerindy-evaluation-9oJ3QbS0SYm7Iag"]
```
