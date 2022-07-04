# Development

### Prepare the environment
For pip/venv:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
_(For newer systems python3 may be referred to as python)_

For Anaconda:
```
conda create --name env pip
conda activate pip
pip install -r requirements.txt
```

## Install deepracer-utils for development
```
python setup.py develop
```
Once finished working, run:
```
python setup.py develop --uninstall
```

See [Python Packaging User Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/#id70) for more info.

## Testing

Run:
```
tox
```
This will package the project, install and run tests.

## Verifying the style guide

Run:
```
pycodestyle
```

## Releasing, Packaging, distribution

Checking the current version:
```
python setup.py version
```

Marking new release:
```
git tag deepracer-utils-version
git push origin deepracer-utils-version
```

Building the package:
```
python setup.py sdist bdist_wheel
```

Uploading to test.pypi.org:
```
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Uploading to pypi.org:
```
python -m twine upload dist/*
```
