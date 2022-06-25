from deepracer.logs.metrics import TrainingMetrics
import numpy as np
import warnings
import pytest
from boto3.exceptions import PythonDeprecationWarning


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

    def test_check_training(self):
        tm = TrainingMetrics(None, fname='./deepracer/logs/sample-console-logs/metrics/training/'
                                         'training-20220611205309-EHNgTNY2T9-77qXhqjBi6A.json')
        training = tm.getTraining()

        assert 28 == max(training['master_iteration'])
        assert 568 == len(training)
        assert np.all([
            "r-i", "round", "iteration", "master_iteration", "episode", "r-e", "worker", "trial", "phase", "reward", "completion", "time", "complete", "start_time",
        ] == training.columns)

    def test_plot(self):
        tm = TrainingMetrics(None, fname='./deepracer/logs/sample-console-logs/metrics/training/'
                                         'training-20220611205309-EHNgTNY2T9-77qXhqjBi6A.json')
        ax = tm.plotProgress()

        assert ax is not None
