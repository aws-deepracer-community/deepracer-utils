from deepracer.logs import DeepRacerLog, AnalysisUtils, SimulationLogsIO, LogType
import numpy as np
import pytest


class TestDeepRacerLog:

    def test_load_model_path(self):
        drl = DeepRacerLog('./deepracer/logs/sample-console-logs')
        drl.load()

        assert LogType.CONSOLE_MODEL_WITH_LOGS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert 4 == len(drl.agent_and_network())
        assert 13 == len(drl.hyperparameters())
        assert 14 == len(drl.action_space())

    def test_dataframe(self):
        drl = DeepRacerLog('./deepracer/logs/sample-console-logs')
        drl.load()
        df = drl.dataframe()

        assert (44247, 20) == df.shape
        assert np.all(['episode', 'steps', 'x', 'y', 'heading', 'steering_angle', 'speed', 'action',
                       'reward', 'done', 'all_wheels_on_track', 'progress', 'closest_waypoint',
                       'track_len', 'tstamp', 'episode_status', 'pause_duration', 'iteration',
                       'worker', 'unique_episode'] == df.columns)

    def test_episode_analysis(self):
        drl = DeepRacerLog('./deepracer/logs/sample-console-logs')
        drl.load()
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df)
        complete_ones = simulation_agg[simulation_agg['progress'] == 100]
        fastest = complete_ones.nsmallest(5, 'time')

        assert (560, 12) == simulation_agg.shape
        assert np.all(['iteration', 'episode', 'steps', 'start_at', 'progress', 'time',
                       'new_reward', 'speed', 'reward', 'time_if_complete',
                       'reward_if_complete', 'quintile'] == simulation_agg.columns)
        assert 432 == fastest.iloc[0, 1]
        assert 213.0 == fastest.iloc[0, 2]
        assert 14.128 == pytest.approx(fastest.iloc[0, 5])

    def test_load_robomaker_logs(self):
        drl = DeepRacerLog('./deepracer/logs/sample-console-logs')
        drl.load_robomaker_logs()

        df = drl.dataframe()

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


class TestEvaluationLogs:
    def test_load_evaluation_logs(self):
        logs = [['deepracer/logs/sample-console-logs/logs/evaluation/'
                 'evaluation-20220612082853-IBZwYd0MRMqgwKlAe7bb0A-robomaker.log', 'log-1'],
                ['deepracer/logs/sample-console-logs/logs/evaluation/'
                 'evaluation-20220612083839-PMfF__s5QJSQT_-E0rEYwg-robomaker.log', 'log-2']]

        bulk = SimulationLogsIO.load_a_list_of_logs(logs)

        assert (1475, 20) == bulk.shape
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

        assert (6, 8) == simulation_agg.shape  # four more episodes in the log
        assert np.all(['stream', 'episode', 'steps', 'start_at', 'progress', 'time', 'speed',
                       'time_if_complete'] == simulation_agg.columns)
        assert 0 == fastest.iloc[0, 1]
        assert 240.0 == fastest.iloc[0, 2]
        assert 15.800 == pytest.approx(fastest.iloc[0, 5])
