import warnings

import matplotlib as plt
import numpy as np
import pytest
from boto3.exceptions import PythonDeprecationWarning

from deepracer.logs import TrainingMetrics


class TestMetrics:

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(tmpdir):
        warnings.filterwarnings("ignore", category=PythonDeprecationWarning)
        yield

    def test_load_metrics(self):
        tm = TrainingMetrics(None, fname='./deepracer/logs/sample-console-logs/metrics/training/'
                                         'training-20220611205309-EHNgTNY2T9-77qXhqjBi6A.json')
        summary = tm.getSummary(method='mean', summary_index=['r-i', 'master_iteration'])
        assert (29, 10) == summary.shape  # CONSOLE_MODEL_WITH_LOGS

    def test_load_multiple_metrics(self):

        tm = TrainingMetrics(None, fname='./deepracer/logs/sample-drfc-3-logs/metrics/'
                                         'TrainingMetrics.json')
        tm.addRound(None, fname='./deepracer/logs/sample-drfc-3-logs/metrics/'
                                'TrainingMetrics_1.json', training_round=1, worker=1)
        summary = tm.getSummary(method='mean', summary_index=['r-i', 'master_iteration'])
        assert (24, 10) == summary.shape  # CONSOLE_MODEL_WITH_LOGS

        training = tm.getTraining()
        assert 479 == len(training)

    def test_check_training(self):
        tm = TrainingMetrics(None, fname='./deepracer/logs/sample-console-logs/metrics/training/'
                                         'training-20220611205309-EHNgTNY2T9-77qXhqjBi6A.json')
        training = tm.getTraining()

        assert 28 == max(training['master_iteration'])
        assert 568 == len(training)
        assert np.all([
            "r-i", "round", "iteration", "master_iteration", "episode", "r-e", "worker", "trial",
            "phase", "reward", "completion", "time", "complete", "start_time",
        ] == training.columns)

    def test_plot(self):
        tm = TrainingMetrics(None, fname='./deepracer/logs/sample-console-logs/metrics/training/'
                                         'training-20220611205309-EHNgTNY2T9-77qXhqjBi6A.json')
        figure = tm.plotProgress(method=["mean", "max"])
        size = figure.get_size_inches()*figure.dpi
        axes = figure.get_axes()

        assert 1200 == size[0]
        assert 500 == size[1]
        assert 2 == len(axes)
