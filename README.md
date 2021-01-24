# deepracer-utils
A set of utilities to take your DeepRacer experience to the next level.

# Add deepracer to AWS CLI and boto3
Make sure you have deepracer-utils >= 0.9 installed.

Then run this to install:
```
python -m deepracer install-cli
```

If you want to ensure you have the latest deepracer service definition, you can run:
```
python -m deepracer install-cli --force
```

To remove deepracer support from aws-cli and boto3, run:
```
python -m deepracer remove-cli
```

## Development

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

### Install deepracer-utils for development
```
python setup.py develop
```
Once finished working, run:
```
python setup.py develop --uninstall
```

See [Python Packaging User Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/#id70) for more info.

### Testing

Run:
```
tox
```
This will package the project, install and run tests.

### Verifying the style guide

Run:
```
pycodestyle
```

### Releasing, Packaging, distribution

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

## License
This project retains the license of the 
[aws-deepracer-workshops](https://github.com/aws-samples/aws-deepracer-workshops)
project which has been forked for the initial Community contributions.
Our understanding is that it is a license more permissive than the MIT license
and allows for removing of the copyright headers. We have decided to preserve
the headers and only add copyright notice for the Community.

## Standards and good practices, contributing
While doing our best to make deepracer-utils an outcome of best practices and standards,
we are using what we learn, as we learn. If you see a solution that would be better to
apply, if you see something that is a risk, do raise it with the Community. Thank you.

We are open to merge requests. Please open an issue first to agree on the outcomes of
your work.

## Contact
You can contact Tomasz Ptak through the Community Slack: http://join.deepracing.io
