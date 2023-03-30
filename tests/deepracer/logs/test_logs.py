import os
import warnings

import numpy as np
import pandas as pd
import pytest
from boto3.exceptions import PythonDeprecationWarning

from deepracer.logs import (AnalysisUtils, DeepRacerLog, LogFolderType,
                            LogType, S3FileHandler, SimulationLogsIO)


class Constants:

    RAW_COLUMNS = ['episode', 'steps', 'x', 'y', 'heading', 'steering_angle', 'speed', 'action',
                   'reward', 'done', 'all_wheels_on_track', 'progress', 'closest_waypoint',
                   'track_len', 'tstamp', 'episode_status', 'pause_duration', 'iteration',
                   'worker', 'unique_episode']

    TRAIN_COLUMNS = ['iteration', 'episode', 'steps', 'start_at', 'progress', 'time', 'dist',
                     'new_reward', 'speed', 'reward', 'time_if_complete',
                     'reward_if_complete', 'quintile', 'complete']

    TRAIN_COLUMNS_UNIQUE = ['iteration', 'unique_episode', 'steps', 'start_at', 'progress', 'time',
                            'dist', 'new_reward', 'speed', 'reward', 'time_if_complete',
                            'reward_if_complete', 'quintile', 'complete']

    EVAL_COLUMNS = ['stream', 'episode', 'steps', 'start_at', 'progress', 'time', 'dist', 'speed',
                    'crashed', 'off_track', 'time_if_complete', 'complete']


class TestTrainingLogs:

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self, tmpdir):
        warnings.filterwarnings("ignore", category=PythonDeprecationWarning)
        yield

    def test_load_model_path(self):
        drl = DeepRacerLog('./deepracer/logs/sample-console-logs')
        drl.load_training_trace(ignore_metadata=True)

        assert LogFolderType.CONSOLE_MODEL_WITH_LOGS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS

    def test_dataframe(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-console-logs')
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert (44247, len(Constants.RAW_COLUMNS)) == df.shape
        assert np.all(Constants.RAW_COLUMNS == df.columns)

    def test_dataframe_load(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-console-logs')
        drl.load(ignore_metadata=True)
        df = drl.dataframe()

        assert (44842, len(Constants.RAW_COLUMNS[:-2])) == df.shape

    def test_episode_analysis(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-console-logs')
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert (560, len(Constants.TRAIN_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS == simulation_agg.columns)
        assert 432 == fastest.iloc[0, 1]
        assert 213.0 == fastest.iloc[0, 2]
        assert 14.128 == pytest.approx(fastest.iloc[0, 5])

    def test_episode_analysis_drfc3_local(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-drfc-3-logs')
        drl.load_training_trace()
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, secondgroup='unique_episode')
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert (690, len(Constants.TRAIN_COLUMNS_UNIQUE)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS_UNIQUE == simulation_agg.columns)
        assert 402 == fastest.iloc[0, 1]
        assert 189.0 == fastest.iloc[0, 2]
        assert 12.548 == pytest.approx(fastest.iloc[0, 5])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_episode_analysis_drfc3_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"),
                           prefix="Analysis-Demo-DRFC-3")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_training_trace()
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, secondgroup='unique_episode')
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert (690, len(Constants.TRAIN_COLUMNS_UNIQUE)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS_UNIQUE == simulation_agg.columns)
        assert 402 == fastest.iloc[0, 1]
        assert 189.0 == fastest.iloc[0, 2]
        assert 12.548 == pytest.approx(fastest.iloc[0, 5])

    def test_episode_analysis_drfc1_local(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-drfc-1-logs')
        drl.load()
        df = drl.dataframe()

        drl.action_space()
        drl.hyperparameters()
        drl.agent_and_network()

        simulation_agg = AnalysisUtils.simulation_agg(df)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert LogFolderType.DRFC_MODEL_SINGLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        print(simulation_agg.columns)
        assert (540, len(Constants.TRAIN_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS == simulation_agg.columns)
        assert 385 == fastest.iloc[0, 1]
        assert 184.0 == fastest.iloc[0, 2]
        assert 12.212 == pytest.approx(fastest.iloc[0, 5])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_episode_analysis_drfc1_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"),
                           prefix="Analysis-Demo-DRFC-1")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_training_trace()
        df = drl.dataframe()

        drl.action_space()
        drl.hyperparameters()
        drl.agent_and_network()

        simulation_agg = AnalysisUtils.simulation_agg(df)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert LogFolderType.DRFC_MODEL_SINGLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert (540, len(Constants.TRAIN_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS == simulation_agg.columns)
        assert 385 == fastest.iloc[0, 1]
        assert 184.0 == fastest.iloc[0, 2]
        assert 12.212 == pytest.approx(fastest.iloc[0, 5])

    def test_load_robomaker_logs(self):
        drl = DeepRacerLog('./deepracer/logs/sample-console-logs')

        with pytest.raises(Exception):
            assert drl.hyperparameters()

        with pytest.raises(Exception):
            assert drl.action_space()

        with pytest.raises(Exception):
            assert drl.agent_and_network()

        drl.load_robomaker_logs()

        df = drl.dataframe()
        print(df)
        simulation_agg = AnalysisUtils.simulation_agg(df)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert 4 == len(drl.agent_and_network())
        assert 13 == len(drl.hyperparameters())
        assert 14 == len(drl.action_space())

        assert (564, len(Constants.TRAIN_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS == simulation_agg.columns)
        assert 432 == fastest.iloc[0, 1]
        assert 213.0 == fastest.iloc[0, 2]
        assert 14.128 == pytest.approx(fastest.iloc[0, 5])


class TestEvaluationLogs:
    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(tmpdir):
        warnings.filterwarnings("ignore", category=PythonDeprecationWarning)
        yield

    def test_load_evaluation_logs(self):
        logs = [['deepracer/logs/sample-console-logs/logs/evaluation/'
                 'evaluation-20220612082853-IBZwYd0MRMqgwKlAe7bb0A-robomaker.log', 'log-1'],
                ['deepracer/logs/sample-console-logs/logs/evaluation/'
                 'evaluation-20220612083839-PMfF__s5QJSQT_-E0rEYwg-robomaker.log', 'log-2']]

        bulk = SimulationLogsIO.load_a_list_of_logs(logs)

        assert (1479, 20) == bulk.shape
        assert np.all(['index', 'iteration', 'episode', 'steps', 'x', 'y', 'yaw',
                       'steering_angle', 'speed', 'action', 'reward', 'done', 'on_track',
                       'progress', 'closest_waypoint', 'track_len', 'tstamp', 'episode_status',
                       'pause_duration', 'stream'] == bulk.columns)

    def test_summarize_evaluation_logs(self):
        logs = [['deepracer/logs/sample-console-logs/logs/evaluation/'
                 'evaluation-20220612082853-IBZwYd0MRMqgwKlAe7bb0A-robomaker.log', 'log-1'],
                ['deepracer/logs/sample-console-logs/logs/evaluation/'
                 'evaluation-20220612083839-PMfF__s5QJSQT_-E0rEYwg-robomaker.log', 'log-2']]

        bulk = SimulationLogsIO.load_a_list_of_logs(logs)

        simulation_agg = AnalysisUtils.simulation_agg(bulk, 'stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 0 == fastest.iloc[0, 1]
        assert 240.0 == fastest.iloc[0, 2]
        assert 15.932 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_analysis(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-console-logs')
        drl.load_evaluation_trace(ignore_metadata=True)
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 15.932 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_laptime_single_laps(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-console-logs')
        drl.load_evaluation_trace(ignore_metadata=True)
        df = drl.dataframe()
        df = df[df['stream'] == '20220612082523']
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)

        total_time = simulation_agg.groupby('stream').agg({'time': ['sum']}).iloc[0,0]
        start_to_finish_time = df['tstamp'].max() - df['tstamp'].min()

        assert (3, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert start_to_finish_time > total_time

    def test_evaluation_laptime_cont_laps(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-drfc-1-logs')
        drl.load_evaluation_trace(ignore_metadata=True)
        df = drl.dataframe()
        df = df[df['stream'] == '20220709200242']
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)

        total_time = simulation_agg.groupby('stream').agg({'time': ['sum']}).iloc[0,0]
        start_to_finish_time = df['tstamp'].max() - df['tstamp'].min()

        assert (3, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert start_to_finish_time == total_time

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_evaluation_analysis_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"),
                           prefix="Analysis-Demo")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert 3 == len(drl.agent_and_network())
        assert 13 == len(drl.hyperparameters())
        assert 14 == len(drl.action_space())

        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 15.932 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_analysis_drfc1_local(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-drfc-1-logs')
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert 3 == len(drl.agent_and_network())
        assert 13 == len(drl.hyperparameters())
        assert 14 == len(drl.action_space())

        assert LogFolderType.DRFC_MODEL_SINGLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert LogType.EVALUATION == drl.active
        assert (9, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 13.405 == pytest.approx(fastest.iloc[0, 5])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_evaluation_analysis_drfc1_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"),
                           prefix="Analysis-Demo-DRFC-1")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert LogFolderType.DRFC_MODEL_SINGLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert LogType.EVALUATION == drl.active
        assert (9, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 13.405 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_analysis_drfc3_local(self):
        drl = DeepRacerLog(model_folder='./deepracer/logs/sample-drfc-3-logs')
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert LogType.EVALUATION == drl.active
        assert (9, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 14.730 == pytest.approx(fastest.iloc[0, 5])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_evaluation_analysis_drfc3_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"),
                           prefix="Analysis-Demo-DRFC-3")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert LogType.EVALUATION == drl.active
        assert (9, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 14.730 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_analysis_robomaker_local(self):
        drl = DeepRacerLog('./deepracer/logs/sample-console-logs')
        drl.load_robomaker_logs(type=LogType.EVALUATION)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert 4 == len(drl.agent_and_network())
        assert 13 == len(drl.hyperparameters())
        assert 14 == len(drl.action_space())

        # four more episodes in the log
        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 0 == fastest.iloc[0, 1]
        assert 240.0 == fastest.iloc[0, 2]
        assert 15.932 == pytest.approx(fastest.iloc[0, 5])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_evaluation_analysis_robomaker_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"),
                           prefix="Analysis-Demo")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_robomaker_logs(type=LogType.EVALUATION)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert 4 == len(drl.agent_and_network())
        assert 13 == len(drl.hyperparameters())
        assert 14 == len(drl.action_space())

        # four more episodes in the log
        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 0 == fastest.iloc[0, 1]
        assert 240.0 == fastest.iloc[0, 2]
        assert 15.932 == pytest.approx(fastest.iloc[0, 5])


class TestLeadershipLogs:
    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(tmpdir):
        warnings.filterwarnings("ignore", category=PythonDeprecationWarning)
        yield

    def test_load_robomaker_logs(self):
        drl = DeepRacerLog('./deepracer/logs/sample-console-logs')
        drl.load_robomaker_logs(type=LogType.LEADERBOARD)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        # four more episodes in the log
        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 2 == fastest.iloc[0, 1]
        assert 234.0 == fastest.iloc[0, 2]
        assert 15.598 == pytest.approx(fastest.iloc[0, 5])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_evaluation_analysis_drfc3_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"),
                           prefix="Analysis-Demo")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_robomaker_logs(type=LogType.LEADERBOARD)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        # four more episodes in the log
        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 2 == fastest.iloc[0, 1]
        assert 234.0 == fastest.iloc[0, 2]
        assert 15.598 == pytest.approx(fastest.iloc[0, 5])
