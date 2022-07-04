import glob
import os
import re
from abc import ABC, abstractmethod
from io import BytesIO

import boto3

from .misc import LogFolderType


class FileHandler(ABC):

    type: LogFolderType = LogFolderType.UNKNOWN_FOLDER
    training_simtrace_path: str = None
    training_simtrace_split: str = None
    training_robomaker_log_path: str = None
    evaluation_simtrace_path: str = None
    evaluation_robomaker_log_path: list = None
    evaluation_simtrace_split: str = None
    leaderboard_robomaker_log_path: list = None

    @abstractmethod
    def list_files(self, filterexp: str = None) -> list:
        pass

    @abstractmethod
    def get_file(self, key: str) -> bytes:
        pass

    @abstractmethod
    def determine_root_folder_type(self) -> LogFolderType:
        pass


class FSFileHandler(FileHandler):

    def __init__(self, model_folder: str, simtrace_path=None, robomaker_log_path=None):
        self.model_folder = model_folder
        self.training_simtrace_path = simtrace_path
        self.training_robomaker_log_path = robomaker_log_path

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

    def determine_root_folder_type(self) -> LogFolderType:

        if os.path.isdir(os.path.join(self.model_folder, "sim-trace")):
            self.type = LogFolderType.CONSOLE_MODEL_WITH_LOGS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = os.path.join(
                    self.model_folder,
                    "sim-trace",
                    "training",
                    "training-simtrace",
                    "*-iteration.csv")
                self.training_simtrace_split = \
                    r'(.*)/training/training-simtrace/(.*)-iteration.csv'
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder,
                    "sim-trace",
                    "evaluation",
                    "*",
                    "evaluation-simtrace",
                    "0-iteration.csv")
                self.evaluation_simtrace_split = \
                    r'.*/evaluation/([0-9]{14})-.*/evaluation-simtrace/(.*)-iteration\.csv'
            if self.training_robomaker_log_path is None:
                self.training_robomaker_log_path = glob.glob(os.path.join(
                    self.model_folder, "**", "training", "*-robomaker.log"))[0]

            if self.evaluation_robomaker_log_path is None:
                self.evaluation_robomaker_log_path = glob.glob(os.path.join(
                    self.model_folder, "**", "evaluation", "*-robomaker.log"))

            if self.leaderboard_robomaker_log_path is None:
                self.leaderboard_robomaker_log_path = glob.glob(os.path.join(
                    self.model_folder, "**", "leaderboard", "*-robomaker.log"))

        elif os.path.isdir(os.path.join(self.model_folder, "training-simtrace")):
            self.type = LogFolderType.DRFC_MODEL_SINGLE_WORKERS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = os.path.join(
                    self.model_folder, "training-simtrace", "*-iteration.csv")
                self.training_simtrace_split = r'(.*)/training-simtrace/(.*)-iteration.csv'
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder,
                    "evaluation-simtrace",
                    "0-iteration.csv")
                self.evaluation_simtrace_split = \
                    r'.*/(.*)/evaluation-simtrace/(.*)-iteration\.csv'

        elif os.path.isdir(os.path.join(self.model_folder, "0")):
            self.type = LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = os.path.join(
                    self.model_folder, "**", "training-simtrace", "*-iteration.csv")
                self.training_simtrace_split = r'.*/(.)/training-simtrace/(.*)-iteration.csv'
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder,
                    "evaluation-simtrace",
                    "0-iteration.csv")
                self.evaluation_simtrace_split = \
                    r'.*/(.*)/evaluation-simtrace/(.*)-iteration\.csv'

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

    def determine_root_folder_type(self) -> LogFolderType:

        if len(self.list_files(filterexp=(self.prefix + r'sim-trace/(.*)'))) > 0:
            self.type = LogFolderType.CONSOLE_MODEL_WITH_LOGS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = self.prefix + \
                    r'sim-trace/training/training-simtrace/(.*)-iteration\.csv'
                self.training_simtrace_split = \
                    r'(.*)/sim-trace/training/training-simtrace/(.*)-iteration.csv'
                self.evaluation_simtrace_path = self.prefix + \
                    r'sim-trace/evaluation/(.*)/evaluation-simtrace/0-iteration\.csv'
                self.evaluation_simtrace_split = \
                    r'.*sim-trace/evaluation/([0-9]{14})-.*/evaluation-simtrace/(.*)-iteration\.csv'
            if self.training_robomaker_log_path is None:
                self.training_robomaker_log_path = self.prefix + \
                    r'logs/training/(.*)-robomaker\.log'
        elif len(self.list_files(filterexp=(self.prefix + r'training-simtrace/(.*)'))) > 0:
            self.type = LogFolderType.DRFC_MODEL_SINGLE_WORKERS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = self.prefix + r'training-simtrace/(.*)-iteration\.csv'
                self.training_simtrace_split = r'(.*)/training-simtrace/(.*)-iteration.csv'
                self.evaluation_simtrace_path = self.prefix + \
                    r'evaluation-simtrace/(.*)-iteration\.csv'
                self.evaluation_simtrace_split = \
                    r'(.*)/evaluation-simtrace/(.*)-iteration\.csv'
        elif len(self.list_files(filterexp=(self.prefix + r'./training-simtrace/(.*)'))) > 0:
            self.type = LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = self.prefix + \
                    r'(.)/training-simtrace/(.*)-iteration\.csv'
                self.training_simtrace_split = r'.*/(.)/training-simtrace/(.*)-iteration.csv'
                self.evaluation_simtrace_path = self.prefix + \
                    r'evaluation-simtrace/(.*)-iteration\.csv'
                self.evaluation_simtrace_split = \
                    r'(.*)/evaluation-simtrace/(.*)-iteration\.csv'
        return self.type
