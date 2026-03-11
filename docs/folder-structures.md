# Supported Folder Structures

deepracer-utils automatically detects the type of model folder you point it at and adapts accordingly. The detection is performed in priority order: when a folder matches more than one heuristic, the first matching rule wins.

## Detection order

```
Does sim-trace/ exist?
 ├── YES: Does sim-trace/training/training-simtrace/ exist?
 │    ├── YES  → CONSOLE_MODEL_WITH_LOGS  (old console format)
 │    └── NO   → DROA_SOLUTION_LOGS       (new console / DROA format)
 └── NO:  Does training-simtrace/ exist at the root?
       ├── YES → DRFC_MODEL_SINGLE_WORKERS
       └── NO:  Does 0/ exist at the root?
             ├── YES → DRFC_MODEL_MULTIPLE_WORKERS
             └── NO:  Does model/ exist at the root?
                   ├── YES → DRFC_MODEL_UPLOAD
                   └── NO  → UNKNOWN_FOLDER (not supported)
```

---

## 1. CONSOLE_MODEL_WITH_LOGS

**Old AWS DeepRacer Console format.** The `training-simtrace/` folder sits directly under `sim-trace/training/`. Log files use the `-robomaker.log` / `-sagemaker.log` naming convention.  Training logs are loaded from the robomaker log rather than from the simtrace CSVs.

```
model-folder/
├── sim-trace/
│   ├── training/
│   │   └── training-simtrace/
│   │       ├── 0-iteration.csv
│   │       ├── 1-iteration.csv
│   │       └── ...
│   └── evaluation/
│       └── {timestamp}-{uuid}/
│           └── evaluation-simtrace/
│               └── 0-iteration.csv
├── logs/
│   ├── training/
│   │   ├── training-{timestamp}-{id}-sagemaker.log
│   │   └── training-{timestamp}-{id}-robomaker.log   ← training data source
│   ├── evaluation/
│   │   └── evaluation-{timestamp}-{uuid}-robomaker.log
│   └── leaderboard/
│       └── leaderboard-{timestamp}-{uuid}-robomaker.log
├── metrics/
│   ├── training/
│   │   └── {timestamp}-{id}.json
│   └── evaluation/
│       └── {timestamp}-{uuid}.json
├── model/
│   └── model_metadata.json
└── ip/
    └── hyperparameters.json
```

**Key properties**

| Property | Value |
|---|---|
| Training source | Robomaker log (`*-robomaker.log`) |
| Evaluation source | `evaluation-simtrace/0-iteration.csv` per run |
| Leaderboard source | `leaderboard/*-robomaker.log` |
| Multi-worker | No |

---

## 2. DROA_SOLUTION_LOGS

**New AWS DeepRacer On AWS (DROA) / Console v2 format.** An ISO 8601 timestamp subdirectory is inserted between `sim-trace/training/` and `training-simtrace/`, and log files use the `-simulation.log` suffix. This type also covers **evaluation-only** downloads where the `training/` subtree is absent.

```
model-folder/
├── sim-trace/
│   ├── training/                                      ← absent in eval-only
│   │   └── {ISO8601}-{name}/
│   │       └── training-simtrace/
│   │           ├── 0-iteration.csv
│   │           ├── 1-iteration.csv
│   │           └── ...
│   └── evaluation/
│       └── {ISO8601}-{name}/
│           └── evaluation-simtrace/
│               └── 0-iteration.csv
├── logs/
│   ├── training/                                      ← absent in eval-only
│   │   └── {ISO8601}-{name}-simulation.log
│   └── evaluation/
│       └── {ISO8601}-{name}-simulation.log
├── metrics/
│   ├── training/                                      ← absent in eval-only
│   │   └── {ISO8601}-{name}.json
│   └── evaluation/
│       └── {ISO8601}-{name}.json
├── model/
│   └── model_metadata.json
└── ip/
    └── hyperparameters.json
```

**Key properties**

| Property | Value |
|---|---|
| Training source | `training-simtrace/*.csv` |
| Evaluation source | `evaluation-simtrace/0-iteration.csv` per run |
| Leaderboard source | `logs/leaderboard/*-simulation.log` |
| Multi-worker | No |

---

## 3. DRFC_MODEL_SINGLE_WORKERS

**DeepRacer for Cloud (DRfC) – single worker.** The `training-simtrace/` folder lives directly at the model root. Evaluation runs appear as sibling folders named `evaluation-{timestamp}/`.

```
model-folder/
├── training-simtrace/
│   ├── 0-iteration.csv
│   ├── 1-iteration.csv
│   └── ...
├── evaluation-{14-digit-timestamp}/
│   └── evaluation-simtrace/
│       └── 0-iteration.csv
├── evaluation-{14-digit-timestamp}/
│   └── evaluation-simtrace/
│       └── 0-iteration.csv
├── metrics/
│   ├── TrainingMetrics.json
│   └── EvaluationMetrics-{timestamp}.json
├── model/
│   └── model_metadata.json
├── ip/
│   └── hyperparameters.json
└── reward_function.py
```

**Key properties**

| Property | Value |
|---|---|
| Training source | `training-simtrace/*.csv` |
| Evaluation source | `evaluation-{timestamp}/evaluation-simtrace/0-iteration.csv` |
| Leaderboard source | Not applicable |
| Multi-worker | No (`worker` column is always `0`) |

---

## 4. DRFC_MODEL_MULTIPLE_WORKERS

**DeepRacer for Cloud (DRfC) – multiple workers.** Each worker's simtrace files are stored in a numbered subdirectory (`0/`, `1/`, `2/`, …). Evaluation and metadata folders remain at the model root.

```
model-folder/
├── 0/
│   └── training-simtrace/
│       ├── 0-iteration.csv
│       ├── 1-iteration.csv
│       └── ...
├── 1/
│   └── training-simtrace/
│       ├── 0-iteration.csv
│       └── ...
├── 2/                               ← if 3 workers were used
│   └── training-simtrace/
│       └── ...
├── evaluation-{14-digit-timestamp}/
│   └── evaluation-simtrace/
│       └── 0-iteration.csv
├── metrics/
│   ├── TrainingMetrics.json
│   ├── TrainingMetrics_1.json       ← per-worker metrics (optional)
│   └── EvaluationMetrics-{timestamp}.json
├── model/
│   └── model_metadata.json
└── ip/
    └── hyperparameters.json
```

**Key properties**

| Property | Value |
|---|---|
| Training source | `{worker}/training-simtrace/*.csv` |
| Evaluation source | `evaluation-{timestamp}/evaluation-simtrace/0-iteration.csv` |
| Leaderboard source | Not applicable |
| Multi-worker | Yes – `worker` column reflects the source worker (0, 1, 2, …) |

---

## 5. DRFC_MODEL_UPLOAD

**DRfC – model upload / evaluation-only.** No `training-simtrace` data is present; training data cannot be loaded. Only evaluation runs are available.

```
model-folder/
├── model/
│   └── model_metadata.json
├── ip/
│   └── hyperparameters.json         ← optional
└── evaluation-{14-digit-timestamp}/
    └── evaluation-simtrace/
        └── 0-iteration.csv
```

**Key properties**

| Property | Value |
|---|---|
| Training source | None |
| Evaluation source | `evaluation-{timestamp}/evaluation-simtrace/0-iteration.csv` |
| Leaderboard source | Not applicable |
| Multi-worker | No |

---

## S3 Support

All folder types are also supported when the model folder resides in an S3 bucket. Use `S3FileHandler` instead of `FSFileHandler`, and point it at the bucket and prefix that corresponds to the model root:

```python
from deepracer.logs import DeepRacerLog
from deepracer.logs.handler import S3FileHandler

fh = S3FileHandler(
    bucket="my-deepracer-bucket",
    prefix="models/my-model",
    region="us-east-1",
)
log = DeepRacerLog(filehandler=fh)
log.load()
```

The same folder-type detection and path resolution logic is applied, with the `prefix` acting as the model root.
