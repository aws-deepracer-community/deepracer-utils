# Modernisation Task List

> **Status key:** ✅ Done · ⏳ Blocked (awaiting input)

## Packaging & Build System

### ✅ 1. Replace `versioneer` with `setuptools-scm`
- `versioneer` 0.18 (~2019) doesn't build cleanly on modern Python and requires a vendored `versioneer.py` in the repo root
- Replace with `setuptools-scm`, the current standard for git-tag-based versioning
- Remove `versioneer.py` and `deepracer/_version.py`
- Update `deepracer/__init__.py` to use `importlib.metadata`

### ✅ 2. Migrate `setup.py` + `setup.cfg` → `pyproject.toml`
- `python3 setup.py build/install` is deprecated (PEP 517/518/621); setuptools now warns on every invocation
- Move all metadata, classifiers, dependencies, and extras into `pyproject.toml` `[project]` table
- Retain `setuptools` as build backend
- Delete `setup.py` and `setup.cfg`
- Replace `MANIFEST.in` with `[tool.setuptools.package-data]` in `pyproject.toml`

### ✅ 3. Update dependency version constraints
| Dependency | Current constraint | Recommended |
|---|---|---|
| `python_requires` | `>=3.8` | `>=3.10` (Ubuntu 22.04 minimum) |
| `numpy` | `>=1.18.0` | `>=1.21.0` |
| `shapely` | `>=1.7.0,<2.0` | `>=2.0.0` |
| `pandas` | `>=1.0.0` | `>=1.3.0` |
| `scikit-learn` | `>=0.22.0` | `>=1.0.0` |
| `matplotlib` | `>=3.1.0` | `>=3.5.0` |
| `joblib` | `>=0.17.0` | `>=1.0.0` |
| `boto3` | `>=1.12.0` | `>=1.26.0` |
| `tensorflow-cpu` (extras) | `>=2.1.0` | `>=2.10.0` |
| `python-resize-image` (extras) | any | **remove** — see task 10 |

### ✅ 4. Update `tox.ini` / CI for Python 3.10 and 3.12
- Extend `tox` envlist to explicitly cover `py310` and `py312` (matching Ubuntu 22.04 and 24.04 system Python)
- Add a GitHub Actions workflow (`.github/workflows/test.yml`) that runs tests against both Ubuntu versions
- Consider replacing `pycodestyle` + `autopep8` with `ruff` (lints and formats, significantly faster)

---

## Deprecations & Removals

### ✅ 5. Remove `boto3_enhancer` and `console` modules
- The AWS DeepRacer Console API that `boto3_enhancer` wraps has been deprecated and is no longer functional
- This removes:
  - `deepracer/boto3_enhancer/` (entire directory, including the bundled `service-2.json` model)
  - `deepracer/console/helper.py` / `deepracer/console/__init__.py` — `ConsoleHelper` depends entirely on `boto3_enhancer`
  - `deepracer/__main__.py` — only exposes `install-cli` / `remove-cli` for the boto3 extension
  - `deepracer/utils.py` — only used to locate `DEEPRACER_UTILS_ROOT` for the boto3 model path
  - Remove `boto3_enhancer` from `deepracer/__init__.py` imports
  - Remove `boto3` from `install_requires` if no other code depends on it (check `S3FileHandler` and `TrainingMetrics`)
  - Update README to remove CLI install/remove instructions

### ✅ 6. Replace `python-resize-image` (unmaintained)
- Last released ~2019; no active maintenance
- Replace the one usage with an equivalent `Pillow` call (`PIL.Image.resize` / `Image.thumbnail`)
- `Pillow` is already a transitive dependency

---

## Compatibility Fixes

### ✅ 7. Fix Shapely 2.0 breaking API changes
- `from shapely.geometry.polygon import LineString` in `deepracer/logs/log_utils.py` and `deepracer/tracks/track_utils.py` — `LineString` lives at `shapely.geometry` in Shapely 2.x (importing from `.polygon` still works but emits a deprecation warning that becomes an error)
- Shapely 2.0 dropped the `cascaded_union` function; verify nothing uses it
- Shapely 2.0 uses numpy-backed geometry; audit any `.coords` manipulation or geometry constructor patterns

### ✅ 8. Fix pandas 2.0 breaking API changes
- `.iteritems()` removed → replace with `.items()`
- `DataFrame.append()` removed → replace with `pd.concat()`
- Silent dtype downcasting removed; explicit `dtype` casts or `pd.to_numeric(..., downcast=...)` calls are required
- Audit all pandas usage in `deepracer/logs/` (`log.py`, `log_utils.py`, `metrics.py`)

### ✅ 9. Update / replace TensorFlow model visualization
- `deepracer/model/visualization.py` uses the TF1 compat API (`tf.disable_v2_behavior()`, `tf.Session`, `tf.GraphDef`) against a frozen `.pb` graph
- Modern DeepRacer models may use a different serialisation format (TF2 SavedModel or ONNX)
- Minimum change: verify the compat.v1 path still works under TF `>=2.10`; gate the import so a missing TF does not break the rest of the package
- Ideal change: add support for loading TF2 SavedModel format; keep frozen-pb path as legacy fallback

---

## DeepRacer Format Updates

### ✅ 10. Add wall-clock column to simtrace parsing
- DeepRacer now optionally writes a wall-clock timestamp column at the end of each simtrace CSV row
- `_COL_NAMES` / `_COL_NAMES_NEW` in `deepracer/logs/log.py` need a `wall_clock` (or equivalent) variant
- The column must be treated as optional (older traces do not have it) — detection should be by column count or header presence
- Update `SimulationLogsIO` parsing logic in `deepracer/logs/log_utils.py` if it parses raw robomaker log lines

### ⏳ 11. Support new console log directory structure
- The AWS DeepRacer Console has introduced a new directory/file layout for downloaded model packages
- **Current status**: exact new structure to be provided — implementation blocked pending specification
- Once provided, update `FSFileHandler.determine_root_folder_type()` and `S3FileHandler` equivalent in `deepracer/logs/handler.py`, adding a new `LogFolderType` enum value if needed
- Add sample log files under `tests/` and corresponding test cases

---

## Documentation

### ✅ 12. Update README installation instructions
- Remove `python3 setup.py build/install` (deprecated)
- Replace with `pip install deepracer-utils` and `pip install -e ".[dev]"` for contributors
- Remove CLI install/remove section (see task 5)
- Update module table to reflect removed modules (`boto3_enhancer`, `console`)
