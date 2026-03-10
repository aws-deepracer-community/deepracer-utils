import json
import logging
import re
from io import BytesIO, TextIOWrapper

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .handler import FileHandler, FSFileHandler
from .log_utils import SimulationLogsIO
from .misc import LogFolderType, LogType
from .stability import SimtraceStabilityAnalyzer


class DeepRacerLog:
    """A class that wraps around a DeepRacer model folder, either in file system or in
    S3 bucket. The class supports both console and DRfC file layouts.

    Methods are exposed to load training logs, evaluation logs or leaderboard submissions.
    """

    # Mapping from the column names used in the simtrace CSV files to the
    # internal names used throughout this library.  Columns not listed here
    # have identical names in both the CSV and the internal representation.
    _CSV_TO_INTERNAL_COLUMNS = {
        "X": "x",
        "Y": "y",
        "yaw": "heading",
        "steer": "steering_angle",
        "throttle": "speed",
    }

    _HYPERPARAM_KEYS = [
        "batch_size",
        "beta_entropy",
        "e_greedy_value",
        "epsilon_steps",
        "exploration_type",
        "loss_type",
        "lr",
        "num_episodes_between_training",
        "num_epochs",
        "stack_size",
        "term_cond_avg_score",
        "term_cond_max_episodes",
    ]

    _MAX_JOBS = 5
    fh: FileHandler = None
    active: LogType = LogType.NOT_DEFINED
    _hyperparameters: dict = None
    _action_space: dict = None
    _agent_and_network: dict = None

    def __init__(
        self,
        model_folder: str = None,
        filehandler: FileHandler = None,
        simtrace_path=None,
        robomaker_log_path=None,
    ):
        """Initializes an object by pointing the class instance to a folder.

         Folder can be on local file system using the model_folder attribute or
        in a custom location through a FileHandler, which also supports S3 locations.

        Args:
            filehandler (FileHandler): Provides an instantiated FileHandler that
                is the proxy for listing and retreiving files.
            model_folder (str): If FileHandler is None, then model_folder will create
                an FSFileHandler pointing to the provided folder.
            simtrace_path (str): Deprecated. Use a custom file handler instead.
            robomaker_log_path (str): Deprecated. Use a custom file handler instead.

        """
        self.model_folder = model_folder

        if simtrace_path is not None:
            raise Exception(
                "Overriding simtrace_path is no longer supported. "
                "Override path using custom File Handler if required."
            )

        if robomaker_log_path is not None:
            raise Exception(
                "Overriding robomaker_log_path is no longer supported. "
                "Override path using custom File Handler if required."
            )

        if filehandler is not None:
            self.fh = filehandler
        else:
            self.fh = FSFileHandler(model_folder)

        self.fh.determine_root_folder_type()

        self.df = None

    def _read_csv(self, path: str, splitRegex, type: LogType = LogType.TRAINING):
        csv_bytes = self.fh.get_file(path)
        df = pd.read_csv(BytesIO(csv_bytes), encoding="utf8")

        # Rename from CSV column names to internal column names.
        df = df.rename(columns=self._CSV_TO_INTERNAL_COLUMNS)

        # Drop the obsolete obstacle counter column if present.
        if "obstacle_crash_counter" in df.columns:
            df = df.drop("obstacle_crash_counter", axis=1)

        # Ensure wall_clock column is present (NaN for traces that predate it).
        if "wall_clock" not in df.columns:
            df["wall_clock"] = float("nan")

        # Ensure pause_duration column is present (NaN for very old traces).
        if "pause_duration" not in df.columns:
            df["pause_duration"] = float("nan")

        path_split = splitRegex.search(path)
        df["iteration"] = int(path_split.groups()[1])

        if self.fh.type == LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS and type == LogType.TRAINING:
            df["worker"] = int(path_split.groups()[0])
        else:
            df["worker"] = 0

        if type == LogType.EVALUATION:
            df["stream"] = path_split.groups()[0]

        if not pd.api.types.is_numeric_dtype(df["action"]):
            df["action"] = -1

        return df

    def load(self, force=False, ignore_metadata=None):
        """Method that loads DeepRacer training trace logs into a dataframe.

        This method is trying to load logs based on folder type.

        """
        if self.fh.type == LogFolderType.CONSOLE_MODEL_WITH_LOGS:
            self.load_robomaker_logs(force=force)
        elif self.fh.type == LogFolderType.DROA_SOLUTION_LOGS:
            self.load_training_trace(force=force, ignore_metadata=ignore_metadata)
        elif self.fh.type == LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS:
            self.load_training_trace(force=force, ignore_metadata=ignore_metadata)
        elif self.fh.type == LogFolderType.DRFC_MODEL_SINGLE_WORKERS:
            self.load_training_trace(force=force, ignore_metadata=ignore_metadata)
        else:
            raise Exception("Unable to load logs from folder.")

    def load_training_trace(self, force: bool = False, ignore_metadata: bool = False):
        """Method that loads DeepRacer training trace logs into a dataframe.

        The method will load in all available workers and iterations from one training run.

        Args:
            force:
                Enables the reloading of logs. If `False` then loading will be blocked
                if the dataframe is already populates.
        """
        self._block_duplicate_load(force)

        if not ignore_metadata:
            self._parse_trace_metadata()

        if self.fh.training_simtrace_path is None:
            raise Exception(
                "Path to training-simtrace not configured. Check FileHandler configuration."
            )

        model_iterations = self.fh.list_files(
            check_exist=True, filterexp=self.fh.training_simtrace_path
        )

        if len(model_iterations) == 0:
            raise Exception("No training-simtrace files found.")

        splitRegex = re.compile(self.fh.training_simtrace_split)

        dfs = Parallel(n_jobs=self._MAX_JOBS, prefer="threads")(
            delayed(self._read_csv)(path, splitRegex, LogType.TRAINING) for path in model_iterations
        )

        if len(dfs) == 0:
            raise Exception("No training-simtrace files loaded.")

        # Merge into single large DataFrame
        df = pd.concat(dfs, ignore_index=True)

        workers_count = df["worker"].astype("int32").max() + 1
        episodes_per_worker = {}
        episodes_until_worker = {0: 0}
        episodes_per_iteration = 0
        for worker in range(workers_count):
            episodes_per_worker[worker] = (
                df[(df["iteration"] == 0) & (df["worker"] == worker)]["episode"].max() + 1
            )
            episodes_until_worker[worker + 1] = (
                episodes_per_worker[worker] + episodes_until_worker[worker]
            )
            episodes_per_iteration += episodes_per_worker[worker]

        logging.debug("workers_count: {}".format(workers_count))
        logging.debug("episodes_per_worker: {}".format(episodes_per_worker))
        logging.debug("episodes_until_worker: {}".format(episodes_until_worker))
        logging.debug("episodes_per_iteration: {}".format(episodes_per_iteration))

        df["unique_episode"] = (
            df["episode"] % np.array([episodes_per_worker[worker] for worker in df["worker"]])
            + np.array([episodes_until_worker[worker] for worker in df["worker"]])
            + df["iteration"] * episodes_per_iteration
        )

        self.df = df.sort_values(["unique_episode", "steps"]).reset_index(drop=True)
        self.active = LogType.TRAINING

    def load_evaluation_trace(self, force: bool = False, ignore_metadata: bool = False):
        """Method that loads DeepRacer evaluation trace logs into a dataframe.

        The method will load in all available evaluations found in a model folder.

        Args:
            force:
                Enables the reloading of logs. If `False` then loading will be blocked
                if the dataframe is already populates.
        """
        self._block_duplicate_load(force)

        if not ignore_metadata:
            self._parse_trace_metadata()

        if self.fh.evaluation_simtrace_path is None:
            raise Exception(
                "Path to evaluation-simtrace not configured. Check FileHandler configuration."
            )

        model_iterations = self.fh.list_files(
            check_exist=True, filterexp=self.fh.evaluation_simtrace_path
        )

        if len(model_iterations) == 0:
            raise Exception("No evaluation-simtrace files found.")

        splitRegex = re.compile(self.fh.evaluation_simtrace_split)

        dfs = Parallel(n_jobs=self._MAX_JOBS, prefer="threads")(
            delayed(self._read_csv)(path, splitRegex, LogType.EVALUATION)
            for path in model_iterations
        )

        if len(dfs) == 0:
            raise Exception("No evaluation-simtrace files loaded.")

        # Merge into single large DataFrame
        df = pd.concat(dfs, ignore_index=True)

        self.df = df.sort_values(["stream", "episode", "steps"]).reset_index(drop=True)
        self.active = LogType.EVALUATION

    def load_robomaker_logs(self, type: LogType = LogType.TRAINING, force: bool = False):
        """Method that loads DeepRacer robomaker log into a dataframe.

        The method will load in all available workers and iterations from one training run.

        Args:
            type:
                By specifying `LogType` as either `TRAINING`, `EVALUATION` or `LEADERBOARD`
                then different logs will be loaded. In the case of `EVALUATION` or `LEADERBOARD`
                multiple logs may be loaded.
            force:
                Enables the reloading of logs. If `False` then loading will be blocked
                if the dataframe is already populates.
        """
        self._block_duplicate_load(force)

        if self.fh.type is not LogFolderType.CONSOLE_MODEL_WITH_LOGS:
            raise Exception("Only supported with LogFolderType.CONSOLE_MODEL_WITH_LOGS")

        if type == LogType.TRAINING:
            raw_data = self.fh.get_file(self.fh.training_robomaker_log_path)

            self._parse_robomaker_metadata(raw_data)

            episodes_per_iteration = self._hyperparameters["num_episodes_between_training"]

            data: list[str] = SimulationLogsIO.load_buffer(
                TextIOWrapper(BytesIO(raw_data), encoding="utf-8")
            )
            self.df = SimulationLogsIO.convert_to_pandas(data, episodes_per_iteration)
            self.active = LogType.TRAINING

        else:
            dfs = []

            if type == LogType.EVALUATION:
                submissions = self.fh.list_files(
                    check_exist=True, filterexp=self.fh.evaluation_robomaker_log_path
                )
                splitRegex = re.compile(self.fh.evaluation_robomaker_split)

            elif type == LogType.LEADERBOARD:
                submissions = self.fh.list_files(
                    check_exist=True, filterexp=self.fh.leaderboard_robomaker_log_path
                )
                splitRegex = re.compile(self.fh.leaderboard_robomaker_log_split)

            for i, log in enumerate(submissions):
                path_split = splitRegex.search(log)
                raw_data = self.fh.get_file(log)

                if i == 0:
                    self._parse_robomaker_metadata(raw_data)

                data = SimulationLogsIO.load_buffer(
                    TextIOWrapper(BytesIO(raw_data), encoding="utf-8")
                )
                dfs.append(SimulationLogsIO.convert_to_pandas(data, stream=path_split.groups()[0]))

            self.df = pd.concat(dfs, ignore_index=True)
            self.active = type

    def _parse_robomaker_metadata(self, raw_data: bytes):

        outside_hyperparams = True
        hyperparameters_string = ""

        data_wrapper = TextIOWrapper(BytesIO(raw_data), encoding="utf-8")

        for line in data_wrapper.readlines():
            if outside_hyperparams:
                if "Using the following hyper-parameters" in line:
                    outside_hyperparams = False
            else:
                hyperparameters_string += line
                if "}" in line:
                    self._hyperparameters = json.loads(hyperparameters_string)
                    break

        data_wrapper.seek(0)

        if self._hyperparameters is None:
            raise Exception("Cound not load hyperparameters. Exiting.")

        for line in data_wrapper.readlines():
            if "ction space from file: " in line:
                self._action_space = json.loads(line.split("file: ")[1].replace("'", '"'))

        data_wrapper.seek(0)

        regex = r"Sensor list (\[[\'a-zA-Z, _-]+\]), network ([a-zA-Z_]+), simapp_version ([\d.]+)"
        agent_and_network = {}
        for line in data_wrapper.readlines():
            if " * /WORLD_NAME: " in line:
                agent_and_network["world"] = line[:-1].split(" ")[-1]
            elif "Sensor list ['" in line:
                m = re.search(regex, line)

                agent_and_network["sensor_list"] = json.loads(m.group(1).replace("'", '"'))
                agent_and_network["network"] = m.group(2)
                agent_and_network["simapp_version"] = m.group(3)

                self._agent_and_network = agent_and_network
                break

        data_wrapper.seek(0)

    def _parse_trace_metadata(self):
        logger = logging.getLogger(__name__)

        if self.fh.model_metadata_path is None:
            logger.debug("model_metadata_path not configured for this handler; skipping.")
        elif self.fh.list_files(check_exist=False, filterexp=self.fh.model_metadata_path):
            try:
                model_metadata: dict = json.load(
                    TextIOWrapper(
                        BytesIO(self.fh.get_file(self.fh.model_metadata_path)), encoding="utf-8"
                    )
                )
                self._action_space = model_metadata["action_space"]
                self._agent_and_network = {}
                self._agent_and_network["sensor_list"] = model_metadata["sensor"]
                self._agent_and_network["network"] = model_metadata["neural_network"]
                self._agent_and_network["simapp_version"] = model_metadata["version"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Could not parse model_metadata.json: %s", e)
        else:
            logger.warning(
                "model_metadata.json not found; action space and agent/network info unavailable."
            )

        if self.fh.hyperparameters_path is None:
            logger.debug("hyperparameters_path not configured for this handler; skipping.")
        elif self.fh.list_files(check_exist=False, filterexp=self.fh.hyperparameters_path):
            try:
                self._hyperparameters = json.load(
                    TextIOWrapper(
                        BytesIO(self.fh.get_file(self.fh.hyperparameters_path)), encoding="utf-8"
                    )
                )
            except json.JSONDecodeError as e:
                logger.warning("Could not parse hyperparameters.json: %s", e)
        else:
            logger.warning("hyperparameters.json not found; hyperparameters unavailable.")

    def dataframe(self):
        """Method that provides the dataframe for analysis of this log."""
        if self.df is None:
            raise Exception("Model not loaded, call load() before requesting a dataframe.")

        return self.df

    def hyperparameters(self) -> dict:
        """Method that provides the hyperparameters for this log."""

        if self._hyperparameters is not None:
            return self._hyperparameters
        else:
            raise Exception("Hyperparameters not yet loaded")

    def action_space(self) -> dict:
        """Method that provides the action space for this log."""
        if self._action_space is not None:
            return self._action_space
        else:
            raise Exception("Action space not yet loaded")

    def agent_and_network(self) -> dict:
        """Method that provides the agent and network information for this log.
        Resulting dictionary includes the name of environment used,
        list of sensors and type of network.
        """

        if self._agent_and_network is not None:
            return self._agent_and_network
        else:
            raise Exception("Agent and Network not yet loaded")

    @property
    def stability(self) -> SimtraceStabilityAnalyzer:
        """A :class:`~deepracer.logs.SimtraceStabilityAnalyzer` bound to the loaded trace.

        The trace must be loaded before accessing this property.  Use whichever
        load method matches the available log format::

            log = DeepRacerLog("./my-model")
            log.load_training_trace()
            df = log.stability.analyze()

        For evaluation stability::

            log.load_evaluation_trace()
            df_eval = log.stability.analyze()

        Raises:
            RuntimeError: If no trace has been loaded yet.
        """
        if self.df is None:
            raise RuntimeError(
                "No trace loaded. Call load(), load_training_trace(), "
                "load_evaluation_trace(), or load_robomaker_logs() first."
            )
        return SimtraceStabilityAnalyzer(self.df)

    def _block_duplicate_load(self, force: bool = False):
        if self.df is not None and not force:
            raise Exception(
                "The dataframe has already been loaded, add force=True"
                + " to your load method to load again"
            )
