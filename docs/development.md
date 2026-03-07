# Development

## Prepare the environment

Requires Python 3.10 or newer (Ubuntu 22.04+).

For pip/venv:
```
python3 -m venv env
source env/bin/activate
pip install -e ".[dev,test]"
```

For Anaconda:
```
conda create --name deepracer-utils python=3.10
conda activate deepracer-utils
pip install -e ".[dev,test]"
```

The `dev` extra installs `black` (formatter) and `setuptools-scm`.  
The `test` extra installs `pytest`, `coverage`, and optional visualization dependencies.

## Install deepracer-utils for development

After activating your virtual environment, run:
```
pip install -e .
```

For the optional visualization features (TensorFlow, OpenCV, Pillow):
```
pip install -e ".[visualization]"
```

See [Python Packaging User Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/#id70) for more info.

## Testing

Run the full test suite via tox (tests on Python 3.10 and 3.12):
```
tox
```

Or run pytest directly in your active environment:
```
pytest tests/
```

## Verifying the style guide

Check formatting:
```
black --check .
```

To auto-fix formatting:
```
black .
```

## Releasing, Packaging, distribution

The package version is derived automatically from git tags using `setuptools-scm`.
There is no separate version file to maintain.

Checking the current version (requires a git tag to be present):
```
python -m setuptools_scm
```

Marking new release:
```
git tag deepracer-utils-<version>
git push origin deepracer-utils-<version>
```

The version number should conform with [PEP 440](https://peps.python.org/pep-0440).

Example: `<version>` can be `0.25` for a release version or `0.25b1` for the first beta of 0.25.

Building the package (requires `build`):
```
pip install build
rm -rf dist/
python -m build
```

Uploading to test.pypi.org:
```
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Uploading to pypi.org:
```
python -m twine upload dist/*
```
