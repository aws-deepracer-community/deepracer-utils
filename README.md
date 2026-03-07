# Deepracer Utilities - Analyzing Your DeepRacer Model
This is a set of utilities that will take your DeepRacer experience to the next level by allowing you to analyze your model, step by step, episode by episode. Only through analyzing what your model does will you be able to write the right reward function, choose the right action space and to tune the hyperparameters!

## Requirements

* Python 3.10 or later (Ubuntu 22.04+)
* See `pyproject.toml` for the full dependency list.

## Installation

Install the latest release from PyPI:
```
pip install deepracer-utils
```

For the optional model visualization features (requires TensorFlow and OpenCV):
```
pip install "deepracer-utils[visualization]"
```

To set up a development environment from a local clone:
```
pip install -e ".[dev,test]"
```

## About the Utilities

The best reference on how to use the utilities can be found in the [deepracer-analysis](https://github.com/aws-deepracer-community/deepracer-analysis) Jupyter notebooks.

An overview of the different modules provided, and the key classes involved:
| Module | Class | Description |
|--------|-------|-------------|
|`deepracer.logs` | `DeepRacerLog` | Points to a DeepRacer model folder (local or S3) and reads simulation trace and robomaker log files.|
|`deepracer.logs` | `AnalysisUtils` | Processes raw log input and summarizes by episode.|
|`deepracer.logs` | `PlottingUtils` | Visualises the track and plots each step in an episode.|
|`deepracer.logs` | `TrainingMetrics` | Reads Metrics data and provides data similar to the training graph in the Console.|
|`deepracer.tracks` | `TrackIO` | Processes track routes (.npy files) and displays waypoints graphically.|
|`deepracer.model` | n/a | Methods to run inference on individual images and to perform visual analysis (requires `visualization` extra).|

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
