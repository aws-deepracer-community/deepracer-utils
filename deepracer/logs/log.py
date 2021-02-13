import glob
import json
import os
import pandas as pd
import re
from joblib import Parallel, delayed

from . import SimulationLogsIO


CONSOLE_MODEL_WITH_LOGS = 0
DRFC_MODEL_SINGLE_WORKERS = 1
DRFC_MODEL_MULTIPLE_WORKERS = 2
UNKNOWN_FOLDER = 3


class DeepRacerLog:
    def __init__(self, model_folder, simtrace_path=None, robomaker_log_path=None):
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

        self._determine_root_folder_type()

        self.df = None

    def load(self, force=False):
        """Method that loads DeepRacer trace logs into a dataframe.
        """
        self._block_duplicate_load(force)

        if self.simtrace_path is None:
            raise Exception(
                "Cannot detect training-simtrace, is model_folder pointing at your model folder?")

        model_iterations = glob.glob(self.simtrace_path)

        def read_csv(path):
            try:
                # TODO: this is a workaround and should be removed when logs are fixed
                df = pd.read_csv(path, names=self.col_names_workaround, header=0)
                df = df.drop("action_b", axis=1)
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(path, names=self.col_names, header=0)
                except pd.errors.ParserError:
                    # Older logs don't have pause_duration, so we're handling this
                    df = pd.read_csv(path, names=self.col_names[:-1], header=0)

            df["iteration"] = int(path.split(os.path.sep)[-1].split("-")[0])
            df["worker"] = int(path.split(os.path.sep)[-3] if self.type ==
                               DRFC_MODEL_MULTIPLE_WORKERS else 0)

            if df.dtypes["action"].name == "object":
                df["action"] = -1

            return df

        dfs = Parallel(n_jobs=-1, prefer="threads")(
            delayed(read_csv)(path) for _, path in enumerate(model_iterations)
        )

        if len(dfs) == 0:
            return

        # Merge into single large DataFrame
        df = pd.concat(dfs, ignore_index=True)

        episodes_per_worker_per_iteration = df[(
            df["iteration"] == 0) & (df["worker"] == 0)]["episode"].max()
        workers_count = df["worker"].max() + 1

        df["unique_episode"] = df["episode"] + df["worker"] * episodes_per_worker_per_iteration + \
            df["iteration"] * episodes_per_worker_per_iteration * workers_count

        self.df = df.sort_values(['unique_episode', 'steps']).reset_index(drop=True)

    def load_robomaker_logs(self, force=False):
        """Method that loads a DeepRacer RoboMaker log into a dataframe.
        """
        self._block_duplicate_load(force)

        self._ensure_robomaker_log_exists()

        episodes_per_iteration = self.hyperparameters()["num_episodes_between_training"]

        self.df = SimulationLogsIO.load_pandas(self.robomaker_log_path, episodes_per_iteration)

    def dataframe(self):
        """Method that provides the dataframe for analysis of this log.
        """
        if self.df is None:
            raise Exception("Model not loaded, call load() before requesting a dataframe.")

        return self.df

    def hyperparameters(self):
        """Method that provides the hyperparameters for this log.
        """
        self._ensure_robomaker_log_exists()

        outside_hyperparams = True
        hyperparameters_string = ""
        with open(self.robomaker_log_path, 'r') as f:
            for line in f:
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
        self._ensure_robomaker_log_exists()

        with open(self.robomaker_log_path, 'r') as f:
            for line in f:
                if "ction space from file: " in line:
                    return json.loads(line.split("file: ")[1].replace("'", '"'))

    def agent_and_network(self):
        """Method that provides the agent and network information for this log.
        Resulting dictionary includes the name of environment used,
        list of sensors and type of network.
        """
        self._ensure_robomaker_log_exists()

        regex = r'Sensor list (\[[\'a-zA-Z, _-]+\]), network ([a-zA-Z_]+), simapp_version ([\d.]+)'

        with open(self.robomaker_log_path, 'r') as f:
            result = {}
            for line in f:
                if " * /WORLD_NAME: " in line:
                    result["world"] = line[:-1].split(" ")[-1]
                elif "Sensor list ['" in line:
                    m = re.search(regex, line)

                    result["sensor_list"] = json.loads(m.group(1).replace("'", '"'))
                    result["network"] = m.group(2)
                    result["simapp_version"] = m.group(3)

                    return result

    def _determine_root_folder_type(self):
        if os.path.isdir(os.path.join(self.model_folder, "sim-trace")):
            self.type = CONSOLE_MODEL_WITH_LOGS
            if self.simtrace_path is None:
                self.simtrace_path = os.path.join(
                    self.model_folder,
                    "sim-trace",
                    "training",
                    "training-simtrace",
                    "*-iteration.csv")
            if self.robomaker_log_path is None:
                self.robomaker_log_path = glob.glob(os.path.join(
                    self.model_folder, "**", "training", "*-robomaker.log"))[0]
        elif os.path.isdir(os.path.join(self.model_folder, "training-simtrace")):
            self.type = DRFC_MODEL_SINGLE_WORKERS
            if self.simtrace_path is None:
                self.simtrace_path = os.path.join(
                    self.model_folder, "training-simtrace", "*-iteration.csv")
        elif os.path.isdir(os.path.join(self.model_folder, "0")):
            self.type = DRFC_MODEL_MULTIPLE_WORKERS
            if self.simtrace_path is None:
                self.simtrace_path = os.path.join(
                    self.model_folder, "**", "training-simtrace", "*-iteration.csv")
        else:
            self.type = UNKNOWN_FOLDER

    def _ensure_robomaker_log_exists(self):
        if self.robomaker_log_path is None or not os.path.isfile(self.robomaker_log_path):
            raise Exception(
                "Cannot detect robomaker log file, is model_folder pointing at your model folder?")

    def _block_duplicate_load(self, force=False):
        if self.df is not None and not force:
            raise Exception(
                "The dataframe has already been loaded, add force=True"
                  + " to your load method to load again")
