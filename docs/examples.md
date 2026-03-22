# Examples

`DeepRacerLog` is the main entry point in deepracer-utils. It supports local folders, S3 buckets, and `.tar.gz` exports from the DeepRacer console via `TarFileHandler`.

## Training

### Local model folder (trace CSV)

Use this when you have an extracted model folder with `sim-trace/` or `training-simtrace/` data.

```python
from deepracer.logs import AnalysisUtils, DeepRacerLog

log = DeepRacerLog(model_folder="./deepracer/logs/sample-console-logs")
log.load_training_trace()
df = log.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df)
```

`df` contains per-step data. `simulation_agg` is per-episode aggregated data.

For continuous-action logs, the `action` column is set to `-1` (no discrete action index). The raw steering and speed values are still available in `steering_angle` and `speed`.

### Local `.tar.gz` export (new)

If you downloaded console logs as a `.tar.gz`, you can read them directly without extracting.

```python
from deepracer.logs import AnalysisUtils, DeepRacerLog, TarFileHandler

fh = TarFileHandler("./deepracer/logs/sample-droa-solution-logs.tar.gz")
log = DeepRacerLog(filehandler=fh)

# New console/DROA archives may not include metadata files.
log.load_training_trace(ignore_metadata=True)
df = log.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df)
```

### Auto-load shortcut

`load()` picks the correct training loader based on detected folder type.

```python
from deepracer.logs import DeepRacerLog, TarFileHandler

log = DeepRacerLog(filehandler=TarFileHandler("./my-model-logs.tar.gz"))
log.load(ignore_metadata=True)
df = log.dataframe()
```

### Verbose output (new)

Pass `verbose=True` to print helpful load progress to stdout.

```python
from deepracer.logs import DeepRacerLog, TarFileHandler

log = DeepRacerLog(
    filehandler=TarFileHandler("./my-model-logs.tar.gz"),
    verbose=True,
)
log.load_training_trace(ignore_metadata=True)
```

Typical output:

```text
Folder type detected: DROA_SOLUTION_LOGS
Loaded training trace: 846 steps, 40 episodes, 2 iterations
```

When `verbose=True`, load methods print a one-line summary:
- `load_training_trace()` -> `Loaded training trace: ...`
- `load_evaluation_trace()` -> `Loaded evaluation trace: ...`
- `load_robomaker_logs()` -> `Loaded robomaker logs: ...`

Each summary reports `steps`, `episodes`, and `iterations`.

### S3 bucket

```python
from deepracer.logs import AnalysisUtils, DeepRacerLog, S3FileHandler

fh = S3FileHandler(bucket="<my_bucket>", prefix="<my_prefix>")
log = DeepRacerLog(filehandler=fh)
log.load_training_trace()
df = log.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df)
```

### Local S3 / MinIO

```python
from deepracer.logs import AnalysisUtils, DeepRacerLog, S3FileHandler

fh = S3FileHandler(
    bucket="<my_bucket>",
    prefix="<my_prefix>",
    profile="<awscli_profile>",
    s3_endpoint_url="<minio_url>",
)
log = DeepRacerLog(filehandler=fh)
log.load_training_trace()
df = log.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df)
```

### Robomaker / simulation logs

For console-style folders, you can also load training from the simulation log file.

```python
from deepracer.logs import AnalysisUtils, DeepRacerLog, TarFileHandler

log = DeepRacerLog(filehandler=TarFileHandler("./my-model-logs.tar.gz"))
log.load_robomaker_logs()
df = log.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df)
```

## Evaluation

Evaluation data can include multiple runs; they are separated with the `stream` column.

```python
from deepracer.logs import AnalysisUtils, DeepRacerLog, TarFileHandler

log = DeepRacerLog(filehandler=TarFileHandler("./my-model-logs.tar.gz"))
log.load_evaluation_trace(ignore_metadata=True)
df = log.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
```

### Evaluation-only archive (new)

Some exports contain only evaluation data. `load_evaluation_trace()` works the same way:

```python
from deepracer.logs import DeepRacerLog, TarFileHandler

log = DeepRacerLog(filehandler=TarFileHandler("./sample-droa-eval-only.tar.gz"))
log.load_evaluation_trace(ignore_metadata=True)
df = log.dataframe()
```

### Evaluation from simulation logs

```python
from deepracer.logs import AnalysisUtils, DeepRacerLog, LogType, TarFileHandler

log = DeepRacerLog(filehandler=TarFileHandler("./my-model-logs.tar.gz"))
log.load_robomaker_logs(type=LogType.EVALUATION)
df = log.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
```

## Leaderboard submissions

Leaderboard submissions are available through simulation log files.

```python
from deepracer.logs import AnalysisUtils, DeepRacerLog, LogType, TarFileHandler

log = DeepRacerLog(filehandler=TarFileHandler("./my-model-logs.tar.gz"))
log.load_robomaker_logs(type=LogType.LEADERBOARD)
df = log.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
```

## Stability analysis (new)

You can analyze step timing stability from loaded simtrace data.

```python
from deepracer.logs import DeepRacerLog, TarFileHandler

log = DeepRacerLog(filehandler=TarFileHandler("./my-model-logs.tar.gz"))
log.load_training_trace(ignore_metadata=True)

stability_df = log.stability.analyze()
print(stability_df[["iteration", "avg_ms", "p95_ms", "rtf"]].head())
```
