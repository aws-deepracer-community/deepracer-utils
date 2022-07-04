import warnings

import numpy as np
import pytest
from boto3.exceptions import PythonDeprecationWarning

from deepracer.console import ConsoleHelper, LeaderboardSubmissionType
from deepracer.logs import AnalysisUtils
from deepracer.logs.metrics import TrainingMetrics


@pytest.mark.skip(reason="Requires AWS access")
class TestConsoleHelper:

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(tmpdir):
        warnings.filterwarnings("ignore", category=PythonDeprecationWarning)
        yield

    def test_find_model(self):

        ch = ConsoleHelper()
        model_arn = ch.find_model("Analysis-Demo")
        assert ('arn:aws:deepracer:us-east-1:180406016328:model/reinforcement_learning/'
                '2977ce21-ef27-4771-9f03-3b9c3cf74fb4' == model_arn)

        training_job = ch.get_training_job(model_arn)
        assert "COMPLETED" == training_job['ActivityJob']['Status']['JobStatus']
        assert model_arn == training_job['ActivityJob']['ModelArn']

    def test_get_metrics(self):

        ch = ConsoleHelper()
        model_arn = ch.find_model("Analysis-Demo")
        training_job = ch.get_training_job(model_arn)
        metrics_url = training_job['ActivityJob']['MetricsPreSignedUrl']

        tm = TrainingMetrics(None, url=metrics_url)
        summary = tm.getSummary()
        assert (29, 10) == summary.shape

    def test_get_training_logs(self):

        ch = ConsoleHelper()
        model_arn = ch.find_model("Analysis-Demo")
        df = ch.get_training_log_robomaker(model_arn)

        simulation_agg = AnalysisUtils.simulation_agg(df)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert (564, 12) == simulation_agg.shape  # four more episodes in the log
        assert np.all(['iteration', 'episode', 'steps', 'start_at', 'progress', 'time',
                       'new_reward', 'speed', 'reward', 'time_if_complete',
                       'reward_if_complete', 'quintile'] == simulation_agg.columns)
        assert 432 == fastest.iloc[0, 1]
        assert 213.0 == fastest.iloc[0, 2]
        assert 14.128 == pytest.approx(fastest.iloc[0, 5])

    def test_get_leaderboard_logs_ranked(self):

        ch = ConsoleHelper()
        df = ch.get_leaderboard_log_robomaker("7bbc2d59-af3c-4e06-ac51-e2c76d9f5734",
                                              select=LeaderboardSubmissionType.RANKED)

        simulation_agg = AnalysisUtils.simulation_agg(df)

        assert (3, 12) == simulation_agg.shape
        assert np.all(['iteration', 'episode', 'steps', 'start_at', 'progress', 'time',
                       'new_reward', 'speed', 'reward', 'time_if_complete',
                       'reward_if_complete', 'quintile'] == simulation_agg.columns)

    def test_get_leaderboard_logs_latest(self):

        ch = ConsoleHelper()
        df = ch.get_leaderboard_log_robomaker("7bbc2d59-af3c-4e06-ac51-e2c76d9f5734",
                                              select=LeaderboardSubmissionType.LATEST)

        simulation_agg = AnalysisUtils.simulation_agg(df)

        assert (3, 12) == simulation_agg.shape
        assert np.all(['iteration', 'episode', 'steps', 'start_at', 'progress', 'time',
                       'new_reward', 'speed', 'reward', 'time_if_complete',
                       'reward_if_complete', 'quintile'] == simulation_agg.columns)
