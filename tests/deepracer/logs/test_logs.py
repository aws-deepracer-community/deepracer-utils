import os
import warnings

import numpy as np
import pytest
from boto3.exceptions import PythonDeprecationWarning

from unittest import mock

from deepracer.logs import (
    AnalysisUtils,
    DeepRacerLog,
    FSFileHandler,
    LogFolderType,
    LogType,
    S3FileHandler,
    SimulationLogsIO,
    TarFileHandler,
)


class Constants:
    RAW_COLUMNS = [
        "episode",
        "steps",
        "x",
        "y",
        "yaw",
        "steering_angle",
        "speed",
        "action",
        "reward",
        "done",
        "on_track",
        "progress",
        "closest_waypoint",
        "track_len",
        "tstamp",
        "episode_status",
        "pause_duration",
        "wall_clock",
        "iteration",
        "worker",
        "unique_episode",
    ]

    TRAIN_COLUMNS = [
        "iteration",
        "episode",
        "steps",
        "start_at",
        "progress",
        "time",
        "dist",
        "new_reward",
        "speed",
        "reward",
        "time_if_complete",
        "reward_if_complete",
        "quintile",
        "complete",
    ]

    TRAIN_COLUMNS_UNIQUE = [
        "iteration",
        "unique_episode",
        "steps",
        "start_at",
        "progress",
        "time",
        "dist",
        "new_reward",
        "speed",
        "reward",
        "time_if_complete",
        "reward_if_complete",
        "quintile",
        "complete",
    ]

    TRAIN_COLUMNS_UNIQUE_PERF = [
        "iteration",
        "unique_episode",
        "steps",
        "start_at",
        "progress",
        "time",
        "dist",
        "new_reward",
        "speed",
        "reward",
        "time_if_complete",
        "reward_if_complete",
        "quintile",
        "complete",
        "step_time_mean",
        "step_time_max",
        "step_time_std",
    ]

    EVAL_COLUMNS = [
        "stream",
        "episode",
        "steps",
        "start_at",
        "progress",
        "time",
        "dist",
        "speed",
        "crashed",
        "off_track",
        "time_if_complete",
        "complete",
    ]


class TestTrainingLogs:
    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self, tmpdir):
        warnings.filterwarnings("ignore", category=PythonDeprecationWarning)
        yield

    def test_load_model_path(self):
        drl = DeepRacerLog("./deepracer/logs/sample-console-logs")
        drl.load_training_trace(ignore_metadata=True)

        assert LogFolderType.CONSOLE_MODEL_WITH_LOGS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS

    def test_dataframe(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-console-logs")
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert (44247, len(Constants.RAW_COLUMNS)) == df.shape
        assert np.all(Constants.RAW_COLUMNS == df.columns)

    def test_dataframe_load(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-console-logs")
        drl.load(ignore_metadata=True)
        df = drl.dataframe()

        assert (44842, len(Constants.RAW_COLUMNS)) == df.shape

    def test_episode_analysis(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-console-logs")
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert (560, len(Constants.TRAIN_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS == simulation_agg.columns)
        assert 432 == fastest.iloc[0, 1]
        assert 213.0 == fastest.iloc[0, 2]
        assert 14.128 == pytest.approx(fastest.iloc[0, 5])

    def test_episode_analysis_drfc3_local(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-drfc-3-logs")
        drl.load_training_trace()
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(
            df, secondgroup="unique_episode", add_perf=True
        )
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")
        print(fastest)

        assert LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert (690, len(Constants.TRAIN_COLUMNS_UNIQUE_PERF)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS_UNIQUE_PERF == simulation_agg.columns)
        assert 402 == fastest["unique_episode"].iloc[0]
        assert 189.0 == fastest["steps"].iloc[0]
        assert 17.12718 == pytest.approx(fastest["dist"].iloc[0], rel=1e-3)
        assert 0.06639 == pytest.approx(fastest["step_time_mean"].iloc[0], rel=1e-3)
        assert 12.548 == pytest.approx(fastest["time"].iloc[0])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_episode_analysis_drfc3_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"), prefix="Analysis-Demo-DRFC-3")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_training_trace()
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, secondgroup="unique_episode")
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert (690, len(Constants.TRAIN_COLUMNS_UNIQUE)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS_UNIQUE == simulation_agg.columns)
        assert 402 == fastest.iloc[0, 1]
        assert 189.0 == fastest.iloc[0, 2]
        assert 12.548 == pytest.approx(fastest.iloc[0, 5])

    def test_episode_analysis_drfc1_local(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-drfc-1-logs")
        drl.load()
        df = drl.dataframe()

        drl.action_space()
        drl.hyperparameters()
        drl.agent_and_network()

        simulation_agg = AnalysisUtils.simulation_agg(df)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert LogFolderType.DRFC_MODEL_SINGLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        print(simulation_agg.columns)
        assert (540, len(Constants.TRAIN_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS == simulation_agg.columns)
        assert 385 == fastest.iloc[0, 1]
        assert 184.0 == fastest.iloc[0, 2]
        assert 12.212 == pytest.approx(fastest.iloc[0, 5])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_episode_analysis_drfc1_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"), prefix="Analysis-Demo-DRFC-1")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_training_trace()
        df = drl.dataframe()

        drl.action_space()
        drl.hyperparameters()
        drl.agent_and_network()

        simulation_agg = AnalysisUtils.simulation_agg(df)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert LogFolderType.DRFC_MODEL_SINGLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert (540, len(Constants.TRAIN_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.TRAIN_COLUMNS == simulation_agg.columns)
        assert 385 == fastest.iloc[0, 1]
        assert 184.0 == fastest.iloc[0, 2]
        assert 12.212 == pytest.approx(fastest.iloc[0, 5])

    def test_load_robomaker_logs(self):
        drl = DeepRacerLog("./deepracer/logs/sample-console-logs")

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
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

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
        logs = [
            [
                "deepracer/logs/sample-console-logs/logs/evaluation/"
                "evaluation-20220612082853-IBZwYd0MRMqgwKlAe7bb0A-robomaker.log",
                "log-1",
            ],
            [
                "deepracer/logs/sample-console-logs/logs/evaluation/"
                "evaluation-20220612083839-PMfF__s5QJSQT_-E0rEYwg-robomaker.log",
                "log-2",
            ],
        ]

        bulk = SimulationLogsIO.load_a_list_of_logs(logs)

        assert (1479, 21) == bulk.shape
        assert np.all(
            [
                "index",
                "iteration",
                "episode",
                "steps",
                "x",
                "y",
                "yaw",
                "steering_angle",
                "speed",
                "action",
                "reward",
                "done",
                "on_track",
                "progress",
                "closest_waypoint",
                "track_len",
                "tstamp",
                "episode_status",
                "pause_duration",
                "wall_clock",
                "stream",
            ]
            == bulk.columns
        )

    def test_summarize_evaluation_logs(self):
        logs = [
            [
                "deepracer/logs/sample-console-logs/logs/evaluation/"
                "evaluation-20220612082853-IBZwYd0MRMqgwKlAe7bb0A-robomaker.log",
                "log-1",
            ],
            [
                "deepracer/logs/sample-console-logs/logs/evaluation/"
                "evaluation-20220612083839-PMfF__s5QJSQT_-E0rEYwg-robomaker.log",
                "log-2",
            ],
        ]

        bulk = SimulationLogsIO.load_a_list_of_logs(logs)

        simulation_agg = AnalysisUtils.simulation_agg(bulk, "stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 0 == fastest.iloc[0, 1]
        assert 240.0 == fastest.iloc[0, 2]
        assert 15.932 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_analysis(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-console-logs")
        drl.load_evaluation_trace(ignore_metadata=True)
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 15.932 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_laptime_single_laps(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-console-logs")
        drl.load_evaluation_trace(ignore_metadata=True)
        df = drl.dataframe()
        df = df[df["stream"] == "20220612082523"]
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)

        total_time = simulation_agg.groupby("stream").agg({"time": ["sum"]}).iloc[0, 0]
        start_to_finish_time = df["tstamp"].max() - df["tstamp"].min()

        assert (3, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert start_to_finish_time > total_time

    def test_evaluation_laptime_cont_laps(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-drfc-1-logs")
        drl.load_evaluation_trace(ignore_metadata=True)
        df = drl.dataframe()
        df = df[df["stream"] == "20220709200242"]
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)

        total_time = simulation_agg.groupby("stream").agg({"time": ["sum"]}).iloc[0, 0]
        start_to_finish_time = df["tstamp"].max() - df["tstamp"].min()

        assert (3, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert start_to_finish_time == total_time

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_evaluation_analysis_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"), prefix="Analysis-Demo")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert 3 == len(drl.agent_and_network())
        assert 13 == len(drl.hyperparameters())
        assert 14 == len(drl.action_space())

        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 15.932 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_analysis_drfc1_local(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-drfc-1-logs")
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

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
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"), prefix="Analysis-Demo-DRFC-1")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert LogFolderType.DRFC_MODEL_SINGLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert LogType.EVALUATION == drl.active
        assert (9, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 13.405 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_analysis_drfc3_local(self):
        drl = DeepRacerLog(model_folder="./deepracer/logs/sample-drfc-3-logs")
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert LogType.EVALUATION == drl.active
        assert (9, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 14.730 == pytest.approx(fastest.iloc[0, 5])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_evaluation_analysis_drfc3_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"), prefix="Analysis-Demo-DRFC-3")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_evaluation_trace()
        df = drl.dataframe()
        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        assert LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS == drl.fh.type  # CONSOLE_MODEL_WITH_LOGS
        assert LogType.EVALUATION == drl.active
        assert (9, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 14.730 == pytest.approx(fastest.iloc[0, 5])

    def test_evaluation_analysis_robomaker_local(self):
        drl = DeepRacerLog("./deepracer/logs/sample-console-logs")
        drl.load_robomaker_logs(type=LogType.EVALUATION)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

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
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"), prefix="Analysis-Demo")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_robomaker_logs(type=LogType.EVALUATION)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

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
        drl = DeepRacerLog("./deepracer/logs/sample-console-logs")
        drl.load_robomaker_logs(type=LogType.LEADERBOARD)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        # four more episodes in the log
        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 2 == fastest.iloc[0, 1]
        assert 234.0 == fastest.iloc[0, 2]
        assert 15.598 == pytest.approx(fastest.iloc[0, 5])

    @pytest.mark.skipif(os.environ.get("TOX_S3_BUCKET", None) is None, reason="Requires AWS access")
    def test_evaluation_analysis_drfc3_s3(self):
        fh = S3FileHandler(bucket=os.environ.get("TOX_S3_BUCKET"), prefix="Analysis-Demo")
        drl = DeepRacerLog(filehandler=fh)
        drl.load_robomaker_logs(type=LogType.LEADERBOARD)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup="stream", is_eval=True)
        complete_ones = simulation_agg[simulation_agg["progress"] == 100]
        fastest = complete_ones.nsmallest(5, "time")

        # four more episodes in the log
        assert (6, len(Constants.EVAL_COLUMNS)) == simulation_agg.shape
        assert np.all(Constants.EVAL_COLUMNS == simulation_agg.columns)
        assert 2 == fastest.iloc[0, 1]
        assert 234.0 == fastest.iloc[0, 2]
        assert 15.598 == pytest.approx(fastest.iloc[0, 5])


class TestDroaSolutionLogs:
    """Tests for the new console log format (DROA_SOLUTION_LOGS).

    Sample data under ``sample-droa-solution-logs/`` contains 2 training iteration files
    and 1 evaluation iteration file.
    """

    # sample-droa-solution-logs holds training iterations 0 and 1 → 846 data rows
    _SAMPLE_DIR = "./deepracer/logs/sample-droa-solution-logs"
    _SAMPLE_TAR = "./deepracer/logs/sample-droa-solution-logs.tar.gz"
    _EXPECTED_ROWS = 846

    # evaluation simtrace: 3 episodes, 1302 data rows
    _EVAL_ONLY_DIR = "./deepracer/logs/sample-droa-eval-only"
    _EVAL_ONLY_TAR = "./deepracer/logs/sample-droa-eval-only.tar.gz"
    _EXPECTED_EVAL_ROWS = 1302

    @pytest.fixture(autouse=True)
    def suppress_warnings(self):
        warnings.filterwarnings("ignore", category=PythonDeprecationWarning)
        yield

    def test_fs_detect_v2_format(self):
        drl = DeepRacerLog(self._SAMPLE_DIR)
        assert LogFolderType.DROA_SOLUTION_LOGS == drl.fh.type

    def test_fs_load_training_trace(self):
        drl = DeepRacerLog(self._SAMPLE_DIR)
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert (self._EXPECTED_ROWS, len(Constants.RAW_COLUMNS)) == df.shape
        assert np.all(Constants.RAW_COLUMNS == df.columns)

    def test_fs_load_shortcut(self):
        """load() on a v2 folder should call load_training_trace automatically."""
        drl = DeepRacerLog(self._SAMPLE_DIR)
        drl.load(ignore_metadata=True)
        df = drl.dataframe()

        assert LogType.TRAINING == drl.active
        assert self._EXPECTED_ROWS == len(df)

    def test_tar_detect_v2_format(self):
        fh = TarFileHandler(self._SAMPLE_TAR)
        fh.determine_root_folder_type()
        assert LogFolderType.DROA_SOLUTION_LOGS == fh.type

    def test_tar_list_files(self):
        fh = TarFileHandler(self._SAMPLE_TAR)
        csvs = fh.list_files(
            filterexp=r"sim-trace/training/[^/]+/training-simtrace/[^/]+-iteration\.csv"
        )
        assert 2 == len(csvs)

    def test_tar_load_training_trace(self):
        fh = TarFileHandler(self._SAMPLE_TAR)
        drl = DeepRacerLog(filehandler=fh)
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert LogFolderType.DROA_SOLUTION_LOGS == drl.fh.type
        assert (self._EXPECTED_ROWS, len(Constants.RAW_COLUMNS)) == df.shape
        assert np.all(Constants.RAW_COLUMNS == df.columns)

    def test_tar_episode_analysis(self):
        fh = TarFileHandler(self._SAMPLE_TAR)
        drl = DeepRacerLog(filehandler=fh)
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        simulation_agg = AnalysisUtils.simulation_agg(df)
        assert len(Constants.TRAIN_COLUMNS) == len(simulation_agg.columns)
        assert np.all(Constants.TRAIN_COLUMNS == simulation_agg.columns)

    def test_fs_load_evaluation_trace(self):
        drl = DeepRacerLog(self._SAMPLE_DIR)
        drl.load_evaluation_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert LogType.EVALUATION == drl.active
        assert (self._EXPECTED_EVAL_ROWS, len(Constants.RAW_COLUMNS)) == df.shape

    def test_tar_load_evaluation_trace(self):
        fh = TarFileHandler(self._SAMPLE_TAR)
        drl = DeepRacerLog(filehandler=fh)
        drl.load_evaluation_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert LogType.EVALUATION == drl.active
        assert (self._EXPECTED_EVAL_ROWS, len(Constants.RAW_COLUMNS)) == df.shape

    def test_eval_only_tar_detect(self):
        """Evaluation-only archive must be detected as DROA_SOLUTION_LOGS."""
        fh = TarFileHandler(self._EVAL_ONLY_TAR)
        fh.determine_root_folder_type()
        assert LogFolderType.DROA_SOLUTION_LOGS == fh.type

    def test_eval_only_tar_load(self):
        fh = TarFileHandler(self._EVAL_ONLY_TAR)
        drl = DeepRacerLog(filehandler=fh)
        drl.load_evaluation_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert LogType.EVALUATION == drl.active
        assert (self._EXPECTED_EVAL_ROWS, len(Constants.RAW_COLUMNS)) == df.shape

    def test_fs_load_training_trace_without_metadata_files(self):
        """load_training_trace() must succeed even when model_metadata.json / hyperparameters.json
        are absent — those files are optional."""
        drl = DeepRacerLog(self._SAMPLE_DIR)
        drl.load_training_trace()  # ignore_metadata defaults to False
        df = drl.dataframe()

        assert (self._EXPECTED_ROWS, len(Constants.RAW_COLUMNS)) == df.shape
        # Metadata is unavailable — accessors raise rather than return stale data.
        with pytest.raises(Exception, match="Hyperparameters not yet loaded"):
            drl.hyperparameters()
        with pytest.raises(Exception, match="Action space not yet loaded"):
            drl.action_space()

    def test_fs_load_evaluation_trace_without_metadata_files(self):
        """load_evaluation_trace() must succeed even when metadata files are absent."""
        drl = DeepRacerLog(self._SAMPLE_DIR)
        drl.load_evaluation_trace()  # ignore_metadata defaults to False
        df = drl.dataframe()

        assert LogType.EVALUATION == drl.active
        assert (self._EXPECTED_EVAL_ROWS, len(Constants.RAW_COLUMNS)) == df.shape

    def test_tstamp_and_wall_clock_always_numeric(self):
        """_read_csv must always produce float64 for tstamp and wall_clock.

        The simtrace CSVs always have a header row.  _read_csv reads them
        natively (no ``names=`` override) so pandas infers dtypes directly
        from the data.  Both tstamp and wall_clock must be float64.
        """
        drl = DeepRacerLog(self._SAMPLE_DIR)
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert df["tstamp"].dtype == np.float64, f"tstamp must be float64, got {df['tstamp'].dtype}"
        assert (
            df["wall_clock"].dtype == np.float64
        ), f"wall_clock must be float64, got {df['wall_clock'].dtype}"

    def test_fs_load_robomaker_logs(self):
        """load_robomaker_logs() must work for DROA_SOLUTION_LOGS.

        The simulation log under logs/training/ uses the same SIM_TRACE_LOG
        format as the console robomaker log, including the extra
        obstacle_crash_counter field (index 17) that precedes wall_clock
        (index 18).
        """
        drl = DeepRacerLog(self._SAMPLE_DIR)
        drl.load_robomaker_logs()

        df = drl.dataframe()
        assert df is not None
        assert len(df) == self._EXPECTED_ROWS

        # Hyperparameters and agent info must be parsed from the log header.
        hp = drl.hyperparameters()
        assert hp["num_episodes_between_training"] == 20
        assert 3 == hp["num_epochs"]

        an = drl.agent_and_network()
        assert an["network"] == "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"
        assert an["simapp_version"] == "6.0"

        # wall_clock must be the actual unix timestamp, not the
        # obstacle_crash_counter (which would be 0).
        assert df["wall_clock"].max() > 1e9

    def test_tar_load_robomaker_logs(self):
        """load_robomaker_logs() must work for a DROA TarFileHandler."""
        fh = TarFileHandler(self._SAMPLE_TAR)
        drl = DeepRacerLog(filehandler=fh)
        drl.load_robomaker_logs()

        df = drl.dataframe()
        assert len(df) == self._EXPECTED_ROWS
        assert drl.hyperparameters()["num_episodes_between_training"] == 20
        assert df["wall_clock"].max() > 1e9


class TestContinuousActionLogs:
    """Tests for DROA logs with a continuous action space.

    In continuous mode the ``action`` column contains a two-element array
    string (e.g. ``[-25.0 1.33]``).  The library normalises it to ``-1``
    (no discrete action index) while still preserving ``steering_angle``
    and ``speed`` from the ``steer``/``throttle`` columns.
    """

    _SAMPLE_TAR = "./deepracer/logs/sample-continous-action-logs.tar.gz"
    _EXPECTED_ROWS = 10693

    @pytest.fixture(autouse=True)
    def suppress_warnings(self):
        warnings.filterwarnings("ignore", category=PythonDeprecationWarning)
        yield

    def test_detect_droa_format(self):
        fh = TarFileHandler(self._SAMPLE_TAR)
        fh.determine_root_folder_type()
        assert LogFolderType.DROA_SOLUTION_LOGS == fh.type

    def test_load_training_trace(self):
        fh = TarFileHandler(self._SAMPLE_TAR)
        drl = DeepRacerLog(filehandler=fh)
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert (self._EXPECTED_ROWS, len(Constants.RAW_COLUMNS)) == df.shape
        assert np.all(Constants.RAW_COLUMNS == df.columns)

    def test_action_is_minus_one_for_continuous_space(self):
        """Continuous-action logs have no discrete action index → action must be -1."""
        fh = TarFileHandler(self._SAMPLE_TAR)
        drl = DeepRacerLog(filehandler=fh)
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert df["action"].unique().tolist() == [-1]

    def test_steering_and_speed_preserved(self):
        """steering_angle and speed must carry the raw continuous values."""
        fh = TarFileHandler(self._SAMPLE_TAR)
        drl = DeepRacerLog(filehandler=fh)
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert df["steering_angle"].dtype == np.float64
        assert df["speed"].dtype == np.float64
        # Sanity: steering spans a range wider than a single discrete value
        assert df["steering_angle"].nunique() > 1
        assert df["speed"].nunique() > 1

    def test_load_robomaker_logs(self):
        """load_robomaker_logs() must handle continuous action entries (action → -1)."""
        fh = TarFileHandler(self._SAMPLE_TAR)
        drl = DeepRacerLog(filehandler=fh)
        drl.load_robomaker_logs()
        df = drl.dataframe()

        assert len(df) == self._EXPECTED_ROWS
        assert df["action"].unique().tolist() == [-1]
        assert df["wall_clock"].max() > 1e9


class TestFSFileHandlerPathNormalization:
    """Tests that FSFileHandler.list_files() normalises path separators.

    On Windows, glob.glob() returns paths with backslash separators (``\\``).
    The split regexes in FSFileHandler all use forward slashes, so without
    normalisation ``re.search(split_regex, path)`` would return ``None`` and
    ``DeepRacerLog._read_csv()`` would raise an ``AttributeError``.
    """

    _SAMPLE_DIR = "./deepracer/logs/sample-console-logs"

    def test_list_files_returns_forward_slashes(self):
        """list_files() must return paths with forward slashes even when glob
        returns backslash-separated paths (simulating Windows behaviour)."""
        fh = FSFileHandler(model_folder=self._SAMPLE_DIR)
        backslash_paths = [
            r"logs\sample-console-logs\training-simtrace\0-iteration.csv",
            r"logs\sample-console-logs\training-simtrace\1-iteration.csv",
        ]
        with mock.patch("deepracer.logs.handler.glob.glob", return_value=backslash_paths):
            result = fh.list_files(filterexp="dummy")
        assert all("/" in p for p in result), "Expected forward slashes in returned paths"
        assert all("\\" not in p for p in result), "Unexpected backslashes in returned paths"

    def test_list_files_forward_slashes_unchanged(self):
        """list_files() must leave forward-slash paths unchanged (non-Windows)."""
        fh = FSFileHandler(model_folder=self._SAMPLE_DIR)
        forward_paths = [
            "logs/sample-console-logs/training-simtrace/0-iteration.csv",
            "logs/sample-console-logs/training-simtrace/1-iteration.csv",
        ]
        with mock.patch("deepracer.logs.handler.glob.glob", return_value=forward_paths):
            result = fh.list_files(filterexp="dummy")
        assert result == forward_paths
