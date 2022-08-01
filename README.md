# Deepracer Utilities - Analyzing Your DeepRacer Model
This is a set of utilities that will take your DeepRacer experience to the next level by allowing you to analyze your model, step by step, episode by episode. Only through analyzing what your model does will you be able to write the right reward function, choose the right action space and to tune the hyperparameters!

## Installation

You can install the latest version of deepracer-utils via pip through
```
pip install deepracer-utils
```
Otherwise you can build your own version with 
```
python3 setup.py build
python3 setup.py install
```

### AWS CLI and boto3 extension
This package contains an extension to the AWS CLI and Boto3 that allows you to interact
with the Deepracer Console through commands starting with `aws deepracer`. For details run
```
aws deepracer help
```

Then run this to install:
```
python -m deepracer install-cli
```

To remove deepracer support from aws-cli and boto3, run:
```
python -m deepracer remove-cli
```

## About the Utilities

The best reference on how to use the utilities can be found in the [deepracer-analysis](https://github.com/aws-deepracer-community/deepracer-analysis) Jupyter notebooks.

An overview of the different modules provided, and the key classes involved:
| Module | Class | Description |
|--------|-------|-------------|
|`deepracer.logs` | DeepRacerLog | Class that is pointed to a Deepracer Model folder, locally or in an S3 bucket, and that reads in and processes trace files from simtrace or robomaker log files.|
|`deepracer.logs` | AnalysisUtils | Class that processes the raw log input and summarizes by episode.|
|`deepracer.logs` | PlottingUtils | Class that visualises the track and plots each step in an episode.|
|`deepracer.logs` | TrainingMetrics | Class that reads in Metrics data and provides data similar to the training graph in the Console.|
|`deepracer.console` | ConsoleHelper | Class that reads out logfiles directly from the console, and together with e.g. TrainingMetrics can be used to visualize training progress in real time.|
|`deepracer.tracks` | TrackIO | Class that processes track routes (.npy files) and displays waypoints graphically.|
|`deepracer.model` | n/a | Methods to run inference on individual images and to perform visual analysis.|
## Other information

* Refer to [development.md](docs/development.md) for instructions on coding standards, unit tests etc.
* Refer to [examples.md](docs/examples.md) for usage guidance.

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
