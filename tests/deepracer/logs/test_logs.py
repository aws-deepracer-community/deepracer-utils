import os
import warnings

import numpy as np
import pytest
from boto3.exceptions import PythonDeprecationWarning

from deepracer.logs import (
    AnalysisUtils,
    DeepRacerLog,
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

        assert (44842, len(Constants.RAW_COLUMNS[:-2])) == df.shape

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

        assert df["tstamp"].dtype == np.float64, (
            f"tstamp must be float64, got {df['tstamp'].dtype}"
        )
        assert df["wall_clock"].dtype == np.float64, (
            f"wall_clock must be float64, got {df['wall_clock'].dtype}"
        )

    def test_load_training_trace_without_action_column(self, tmp_path):
        """Regression: _read_csv must not raise KeyError when the simtrace CSV
        has no 'action' column (older simulator versions omit it).
        The column should be filled with -1.
        """
        # Build a minimal DROA_SOLUTION_LOGS folder structure.
        csv_dir = (
            tmp_path
            / "sim-trace"
            / "training"
            / "2024-01-01T00:00:00.000Z-test"
            / "training-simtrace"
        )
        csv_dir.mkdir(parents=True)
        # Write a one-row CSV without the 'action' column.
        csv_path = csv_dir / "0-iteration.csv"
        csv_path.write_text(
            "episode,steps,X,Y,yaw,steer,throttle,reward,done,"
            "all_wheels_on_track,progress,closest_waypoint,track_len,"
            "tstamp,episode_status,pause_duration,wall_clock\n"
            "0,1,0.1,0.2,0.0,0.0,1.0,0.5,False,True,1.0,0,60.0,"
            "100.0,in_progress,0.0,1000.0\n"
            "0,2,0.1,0.2,0.0,0.0,1.0,0.5,True,True,2.0,1,60.0,"
            "100.1,finish,0.0,1000.1\n",
            encoding="utf-8",
        )

        drl = DeepRacerLog(str(tmp_path))
        drl.load_training_trace(ignore_metadata=True)
        df = drl.dataframe()

        assert "action" in df.columns, "action column must be present after load"
        assert (df["action"] == -1).all(), "missing action column must be filled with -1"
