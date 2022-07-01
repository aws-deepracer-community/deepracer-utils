import glob
import json
import os
import pandas as pd
import numpy as np
import logging
import re
from io import BytesIO, StringIO, TextIOWrapper
from enum import Enum
from joblib import Parallel, delayed

import boto3
from . import SimulationLogsIO

from abc import ABC, abstractmethod
from typing import Tuple


class LogType(Enum):
    CONSOLE_MODEL_WITH_LOGS = 0
    DRFC_MODEL_SINGLE_WORKERS = 1
    DRFC_MODEL_MULTIPLE_WORKERS = 2
    UNKNOWN_FOLDER = 3


class FileHandler(ABC):

    type: LogType = LogType.UNKNOWN_FOLDER
    simtrace_path: str = None
    robomaker_log_path: str = None

    @abstractmethod
    def list_files(self, filterexp: str = None) -> list:
        pass

    @abstractmethod
    def get_file(self, key: str) -> bytes:
        pass

    @abstractmethod
    def determine_root_folder_type(self) -> LogType:
        pass


class FSFileHandler(FileHandler):

    def __init__(self, model_folder: str, simtrace_path=None, robomaker_log_path=None):
        self.model_folder = model_folder
        self.simtrace_path = simtrace_path
        self.robomaker_log_path = robomaker_log_path

    def list_files(self, filterexp: str = None) -> list:
        if filterexp is None:
            return glob.glob(self.model_folder)
        else:
            return glob.glob(filterexp)

    def get_file(self, key: str) -> bytes:
        bytes_io: BytesIO = None
        with open(key, 'rb') as fh:
            bytes_io = BytesIO(fh.read())
        return bytes_io.getvalue()

    def determine_root_folder_type(self) -> LogType:

        if os.path.isdir(os.path.join(self.model_folder, "sim-trace")):
            self.type = LogType.CONSOLE_MODEL_WITH_LOGS
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
            self.type = LogType.DRFC_MODEL_SINGLE_WORKERS
            if self.simtrace_path is None:
                self.simtrace_path = os.path.join(
                    self.model_folder, "training-simtrace", "*-iteration.csv")
        elif os.path.isdir(os.path.join(self.model_folder, "0")):
            self.type = LogType.DRFC_MODEL_MULTIPLE_WORKERS
            if self.simtrace_path is None:
                self.simtrace_path = os.path.join(
                    self.model_folder, "**", "training-simtrace", "*-iteration.csv")

        return self.type


class S3FileHandler(FileHandler):

    def __init__(self, bucket: str, prefix: str = None,
                 s3_endpoint_url: str = None, region: str = None, profile: str = None,
                 ):
        if profile is not None:
            session = boto3.session.Session(profile_name=profile)
            self.s3 = session.resource("s3", endpoint_url=s3_endpoint_url, region_name=region)
        else:
            self.s3 = boto3.resource("s3", endpoint_url=s3_endpoint_url, region_name=region)

        self.bucket = bucket

        if (prefix[-1:] == "/"):
            self.prefix = prefix
        else:
            self.prefix = "{}/".format(prefix)

    def list_files(self, filterexp: str = None) -> list:
        files = []

        bucket_obj = self.s3.Bucket(self.bucket)
        for objects in bucket_obj.objects.filter(Prefix=self.prefix):
            files.append(objects.key)

        if filterexp is not None:
            return [x for x in files if re.match(filterexp, x)]
        else:
            return files

    def get_file(self, key: str) -> bytes:

        bytes_io = BytesIO()
        self.s3.Object(self.bucket, key).download_fileobj(bytes_io)
        return bytes_io.getvalue()

    def determine_root_folder_type(self) -> LogType:

        if len(self.list_files(filterexp=(self.prefix + r'sim-trace/(.*)'))) > 0:
            self.type = LogType.CONSOLE_MODEL_WITH_LOGS
            if self.simtrace_path is None:
                self.simtrace_path = self.prefix + \
                    r'sim-trace/training/training-simtrace/(.*)-iteration\.csv'
            if self.robomaker_log_path is None:
                self.robomaker_log_path = self.prefix + r'logs/training/(.*)-robomaker\.log'
        elif len(self.list_files(filterexp=(self.prefix + r'training-simtrace/(.*)'))) > 0:
            self.type = LogType.DRFC_MODEL_SINGLE_WORKERS
            if self.simtrace_path is None:
                self.simtrace_path = self.prefix + r'training-simtrace/(.*)-iteration\.csv'
        elif len(self.list_files(filterexp=(self.prefix + r'./training-simtrace/(.*)'))) > 0:
            self.type = LogType.DRFC_MODEL_MULTIPLE_WORKERS
            if self.simtrace_path is None:
                self.simtrace_path = self.prefix + r'./training-simtrace/(.*)-iteration\.csv'

        return self.type


class DeepRacerLog:

    fh: FileHandler = None

    def __init__(self, model_folder=None, simtrace_path=None, robomaker_log_path=None,
                 filehandler: FileHandler = None):
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

    def load(self, force=False):
        """Method that loads DeepRacer trace logs into a dataframe.
        """
        self._block_duplicate_load(force)

        if self.fh.simtrace_path is None:
            raise Exception(
                "Cannot detect training-simtrace, is model_folder pointing at your model folder?")

        model_iterations = self.fh.list_files(filterexp=self.fh.simtrace_path)

        def read_csv(path: str):
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
            df["iteration"] = int(path.split(os.path.sep)[-1].split("-")[0])
            df["worker"] = int(path.split(os.path.sep)[-3] if self.fh.type ==
                               LogType.DRFC_MODEL_MULTIPLE_WORKERS else 0)
            if df.dtypes["action"].name == "object":
                df["action"] = -1

            return df

        dfs = Parallel(n_jobs=-1, prefer="threads")(
            delayed(read_csv)(path) for path in model_iterations
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
#        logging.debug("episodes_until_worker: {}".format(episodes_until_worker))
        logging.debug("episodes_per_iteration: {}".format(episodes_per_iteration))

        df["unique_episode"] = df["episode"] % np.array([episodes_per_worker[worker]
                                                         for worker in df["worker"]]) + \
            np.array([episodes_until_worker[worker] for worker in df["worker"]]) + \
            df["iteration"] * episodes_per_iteration

        self.df = df.sort_values(['unique_episode', 'steps']).reset_index(drop=True)

    def load_robomaker_logs(self, force=False):
        """Method that loads a DeepRacer RoboMaker log into a dataframe.
        """
        self._block_duplicate_load(force)

        self._ensure_robomaker_log_exists()

        episodes_per_iteration = self.hyperparameters()["num_episodes_between_training"]

        data = SimulationLogsIO.load_buffer(TextIOWrapper(
            BytesIO(self.fh.get_file(self.fh.robomaker_log_path)), encoding='utf-8'))
        self.df = SimulationLogsIO.convert_to_pandas(data, episodes_per_iteration)

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

        text_io = TextIOWrapper(BytesIO(self.fh.get_file(self.fh.robomaker_log_path)),
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
        self._ensure_robomaker_log_exists()

        text_io = TextIOWrapper(BytesIO(self.fh.get_file(
            self.fh.robomaker_log_path)), encoding='utf-8')

        for line in text_io.readlines():
            if "ction space from file: " in line:
                return json.loads(line.split("file: ")[1].replace("'", '"'))

    def agent_and_network(self):
        """Method that provides the agent and network information for this log.
        Resulting dictionary includes the name of environment used,
        list of sensors and type of network.
        """
        self._ensure_robomaker_log_exists()

        regex = r'Sensor list (\[[\'a-zA-Z, _-]+\]), network ([a-zA-Z_]+), simapp_version ([\d.]+)'

        text_io = TextIOWrapper(BytesIO(self.fh.get_file(
            self.fh.robomaker_log_path)), encoding='utf-8')
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

    def _ensure_robomaker_log_exists(self):
        if self.fh.robomaker_log_path is None or \
                len(self.fh.list_files(self.fh.robomaker_log_path)) == 0:
            raise Exception(
                "Cannot detect robomaker log file, is model_folder pointing at your model folder?")

    def _block_duplicate_load(self, force=False):
        if self.df is not None and not force:
            raise Exception(
                "The dataframe has already been loaded, add force=True"
                + " to your load method to load again")
