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


class DeepRacerLog:

    _MAX_JOBS = 5
    fh: FileHandler = None
    active: LogType = LogType.NOT_DEFINED

    def __init__(self, model_folder=None, filehandler: FileHandler = None,
                 simtrace_path=None, robomaker_log_path=None):
        # Column names we support in the CSV file.
        self.col_names = [
            "episode",
            "steps",
            "x",
            "y",
            "heading",
            "steering_angle",
            "speed",
            "action",
            "reward",
            "done",
            "all_wheels_on_track",
            "progress",
            "closest_waypoint",
            "track_len",
            "tstamp",
            "episode_status",
            "pause_duration"
        ]
        # TODO Column names as a workaround for an excess comma in the CSV file
        self.col_names_workaround = [
            "episode",
            "steps",
            "x",
            "y",
            "heading",
            "steering_angle",
            "speed",
            "action",
            "action_b",
            "reward",
            "done",
            "all_wheels_on_track",
            "progress",
            "closest_waypoint",
            "track_len",
            "tstamp",
            "episode_status",
            "pause_duration"
        ]
        self.hyperparam_keys = [
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
            "term_cond_max_episodes"
        ]

        self.model_folder = model_folder
        self.simtrace_path = simtrace_path
        self.robomaker_log_path = robomaker_log_path

        if filehandler is not None:
            self.fh = filehandler
        else:
            self.fh = FSFileHandler(model_folder)

        self.fh.determine_root_folder_type()

        self.df = None

    def read_csv(self, path: str, splitRegex, type: LogType = LogType.TRAINING):
        try:
            csv_bytes = self.fh.get_file(path)
            # TODO: this is a workaround and should be removed when logs are fixed
            df = pd.read_csv(BytesIO(csv_bytes), encoding='utf8',
                             names=self.col_names_workaround, header=0)
            df = df.drop("action_b", axis=1)
        except pd.errors.ParserError:
            try:
                df = pd.read_csv(BytesIO(csv_bytes), names=self.col_names, header=0)
            except pd.errors.ParserError:
                # Older logs don't have pause_duration, so we're handling this
                df = pd.read_csv(BytesIO(csv_bytes), names=self.col_names[:-1], header=0)

        path_split = splitRegex.search(path)
        df["iteration"] = int(path_split.groups()[1])

        if (self.fh.type == LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS and type == LogType.TRAINING):
            df["worker"] = int(path_split.groups()[0])
        else:
            df["worker"] = 0

        if (type == LogType.EVALUATION):
            df["stream"] = path_split.groups()[0]

        if df.dtypes["action"].name == "object":
            df["action"] = -1

        return df

    def load(self, force=False):
        """ Method that loads DeepRacer training trace logs into a dataframe.
            This method is deprecated, use load_training_trace.
        """
        self.load_training_trace(force)

    def load_training_trace(self, force=False):
        """ Method that loads DeepRacer training trace logs into a dataframe.
            The method will load in all available workers and iterations from one training run.
        """
        self._block_duplicate_load(force)

        if self.fh.training_simtrace_path is None:
            raise Exception(
                "Cannot detect training-simtrace, is model_folder pointing at your model folder?")

        model_iterations = self.fh.list_files(filterexp=self.fh.training_simtrace_path)
        splitRegex = re.compile(self.fh.training_simtrace_split)

        dfs = Parallel(n_jobs=self._MAX_JOBS, prefer="threads")(
            delayed(self.read_csv)(path, splitRegex, LogType.TRAINING) for path in model_iterations
        )

        if len(dfs) == 0:
            return

        # Merge into single large DataFrame
        df = pd.concat(dfs, ignore_index=True)

        workers_count = df["worker"].astype('int32').max() + 1
        episodes_per_worker = {}
        episodes_until_worker = {0: 0}
        episodes_per_iteration = 0
        for worker in range(workers_count):
            episodes_per_worker[worker] = df[(df["iteration"] == 0) & (
                df["worker"] == worker)]["episode"].max() + 1
            episodes_until_worker[worker + 1] = episodes_per_worker[worker] + \
                episodes_until_worker[worker]
            episodes_per_iteration += episodes_per_worker[worker]

        logging.debug("workers_count: {}".format(workers_count))
        logging.debug("episodes_per_worker: {}".format(episodes_per_worker))
        logging.debug("episodes_until_worker: {}".format(episodes_until_worker))
        logging.debug("episodes_per_iteration: {}".format(episodes_per_iteration))

        df["unique_episode"] = df["episode"] % np.array([episodes_per_worker[worker]
                                                         for worker in df["worker"]]) + \
            np.array([episodes_until_worker[worker] for worker in df["worker"]]) + \
            df["iteration"] * episodes_per_iteration

        self.df = df.sort_values(['unique_episode', 'steps']).reset_index(drop=True)
        self.active = LogType.TRAINING

    def load_evaluation_trace(self, force=False):
        """ Method that loads DeepRacer training trace logs into a dataframe.
            The method will load in all available workers and iterations from one training run.
        """
        self._block_duplicate_load(force)

        if self.fh.training_simtrace_path is None:
            raise Exception(
                "Cannot detect training-simtrace, is model_folder pointing at your model folder?")

        model_iterations = self.fh.list_files(filterexp=self.fh.evaluation_simtrace_path)
        splitRegex = re.compile(self.fh.evaluation_simtrace_split)

        dfs = Parallel(n_jobs=self._MAX_JOBS, prefer="threads")(
            delayed(self.read_csv)(path, splitRegex, LogType.EVALUATION)
            for path in model_iterations
        )

        if len(dfs) == 0:
            return

        # Merge into single large DataFrame
        df = pd.concat(dfs, ignore_index=True)

        self.df = df.sort_values(['stream', 'episode', 'steps']).reset_index(drop=True)
        self.active = LogType.EVALUATION

    def load_robomaker_logs(self, type: LogType = LogType.TRAINING, force=False):
        """Method that loads a single DeepRacer RoboMaker log into a dataframe.
        """
        self._block_duplicate_load(force)

        self._ensure_file_exists()

        if type == LogType.TRAINING:
            episodes_per_iteration = self.hyperparameters()["num_episodes_between_training"]

            data = SimulationLogsIO.load_buffer(TextIOWrapper(
                BytesIO(self.fh.get_file(self.fh.training_robomaker_log_path)), encoding='utf-8'))
            self.df = SimulationLogsIO.convert_to_pandas(data, episodes_per_iteration)
            self.active = LogType.TRAINING
        else:
            if type == LogType.EVALUATION:
                self.active = LogType.EVALUATION

            elif type == LogType.LEADERBOARD:
                self.active = LogType.LEADERBOARD

    def dataframe(self):
        """Method that provides the dataframe for analysis of this log.
        """
        if self.df is None:
            raise Exception("Model not loaded, call load() before requesting a dataframe.")

        return self.df

    def hyperparameters(self):
        """Method that provides the hyperparameters for this log.
        """
        self._ensure_file_exists()

        outside_hyperparams = True
        hyperparameters_string = ""

        text_io = TextIOWrapper(BytesIO(self.fh.get_file(self.fh.training_robomaker_log_path)),
                                encoding='utf-8')

        for line in text_io.readlines():
            if outside_hyperparams:
                if "Using the following hyper-parameters" in line:
                    outside_hyperparams = False
            else:
                hyperparameters_string += line
                if "}" in line:
                    break

        return json.loads(hyperparameters_string)

    def action_space(self):
        """Method that provides the action space for this log.
        """
        self._ensure_file_exists()

        text_io = TextIOWrapper(BytesIO(self.fh.get_file(
            self.fh.training_robomaker_log_path)), encoding='utf-8')

        for line in text_io.readlines():
            if "ction space from file: " in line:
                return json.loads(line.split("file: ")[1].replace("'", '"'))

    def agent_and_network(self):
        """Method that provides the agent and network information for this log.
        Resulting dictionary includes the name of environment used,
        list of sensors and type of network.
        """
        self._ensure_file_exists()

        regex = r'Sensor list (\[[\'a-zA-Z, _-]+\]), network ([a-zA-Z_]+), simapp_version ([\d.]+)'

        text_io = TextIOWrapper(BytesIO(self.fh.get_file(
            self.fh.training_robomaker_log_path)), encoding='utf-8')
        result = {}
        for line in text_io.readlines():
            if " * /WORLD_NAME: " in line:
                result["world"] = line[:-1].split(" ")[-1]
            elif "Sensor list ['" in line:
                m = re.search(regex, line)

                result["sensor_list"] = json.loads(m.group(1).replace("'", '"'))
                result["network"] = m.group(2)
                result["simapp_version"] = m.group(3)

                return result

    def _ensure_file_exists(self, file: str = None):
        if self.fh.training_robomaker_log_path is None or \
                len(self.fh.list_files(self.fh.training_robomaker_log_path)) == 0:
            raise Exception(
                "Cannot detect robomaker log file, is model_folder pointing at your model folder?")

    def _block_duplicate_load(self, force=False):
        if self.df is not None and not force:
            raise Exception(
                "The dataframe has already been loaded, add force=True"
                + " to your load method to load again")
