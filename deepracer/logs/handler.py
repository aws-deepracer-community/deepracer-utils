import glob
import os
import re
from abc import ABC, abstractmethod
from io import BytesIO

import boto3

from .misc import LogFolderType


class FileHandler(ABC):
    """An abstract base class that encapsulates the interaction with a model storage.

    Currently implemented trough `FSFileHandler` and `S3FileHandler`.

    Methods are exposed to detect what kind of model folder is present, to list files
    and to read files into a buffer.
    """
    type: LogFolderType = LogFolderType.UNKNOWN_FOLDER
    training_simtrace_path: str = None
    training_simtrace_split: str = None
    training_robomaker_log_path: str = None
    evaluation_simtrace_path: str = None
    evaluation_robomaker_log_path: str = None
    evaluation_robomaker_split: str = None
    evaluation_simtrace_split: str = None
    leaderboard_robomaker_log_path: str = None
    leaderboard_robomaker_log_split: str = None
    model_metadata_path: str = None
    hyperparameters_path: str = None

    @abstractmethod
    def list_files(self, filterexp: str = None, check_exist: bool = False) -> list:
        pass

    @abstractmethod
    def get_file(self, key: str) -> bytes:
        pass

    @abstractmethod
    def determine_root_folder_type(self) -> LogFolderType:
        pass


class FSFileHandler(FileHandler):
    """An implementation of the FileHandler interface that interacts with the local filesystem.

    Methods are exposed to detect what kind of model folder is present, to list files
    and to read files into a buffer.
    """
    def __init__(self, model_folder: str, simtrace_path=None, robomaker_log_path=None):
        self.model_folder = model_folder
        self.training_simtrace_path = simtrace_path
        self.training_robomaker_log_path = robomaker_log_path

    def list_files(self, filterexp: str = None, check_exist: bool = False) -> list:
        """Lists a set of files.

        If `filterexp` is provided then a search is performed using provided pattern.
        Otherwise all files in the `model_folder` is returned.

        Args:
            filterexp:
                Filter expression. Typically the values for `training_simtrace_path`,
                `evaluation_simtrace_path` etc. are provided.
            check_exist:
                If `True` then raise Exception if no files are found.
                Otherwise return empty list.

        Returns:
            A list of files.
        """
        if check_exist and (filterexp is None and self.model_folder is None):
            raise Exception("File path is not defined.")

        if filterexp is None:
            return_files = glob.glob(self.model_folder)
        else:
            return_files = glob.glob(filterexp)

        if len(return_files) > 0:
            return return_files
        else:
            if check_exist:
                raise Exception("No files found in {} or {}".format(self.model_folder, filterexp))
            else:
                return []

    def get_file(self, key: str) -> bytes:
        """Downloads a given file as byte array.

        Args:
            key:
                Path to a file on the filesystem.

        Returns:
            A bytes object containing the file.
        """

        bytes_io: BytesIO = None
        with open(key, 'rb') as fh:
            bytes_io = BytesIO(fh.read())
        return bytes_io.getvalue()

    def determine_root_folder_type(self) -> LogFolderType:

        if os.path.isdir(os.path.join(self.model_folder, "sim-trace")):
            self.type = LogFolderType.CONSOLE_MODEL_WITH_LOGS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = os.path.join(
                    self.model_folder, "sim-trace", "training", "training-simtrace",
                    "*-iteration.csv")
                self.training_simtrace_split = r'(.*)/training/training-simtrace/(.*)-iteration.csv'
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder, "sim-trace", "evaluation", "*", "evaluation-simtrace",
                    "0-iteration.csv")
                self.evaluation_simtrace_split = \
                    r'.*/evaluation/([0-9]{14})-.*/evaluation-simtrace/(.*)-iteration\.csv'
            if self.training_robomaker_log_path is None:
                self.training_robomaker_log_path = glob.glob(os.path.join(
                    self.model_folder, "**", "training", "*-robomaker.log"))[0]

            if self.evaluation_robomaker_log_path is None:
                self.evaluation_robomaker_log_path = os.path.join(
                    self.model_folder, "**", "evaluation", "*-robomaker.log")

            if self.evaluation_robomaker_split is None:
                self.evaluation_robomaker_split = \
                    r'.*/evaluation/evaluation-([0-9]{14})-(.*)-robomaker.log'

            if self.leaderboard_robomaker_log_path is None:
                self.leaderboard_robomaker_log_path = os.path.join(
                    self.model_folder, "**", "leaderboard", "*-robomaker.log")

            if self.leaderboard_robomaker_log_split is None:
                self.leaderboard_robomaker_log_split = \
                    r'.*/leaderboard/leaderboard-([0-9]{14})-(.*)-robomaker.log'

            if self.model_metadata_path is None:
                self.model_metadata_path = os.path.join(
                    self.model_folder, "model", "model_metadata.json")

            if self.hyperparameters_path is None:
                self.hyperparameters_path = os.path.join(
                    self.model_folder, "ip", "hyperparameters.json")

        elif os.path.isdir(os.path.join(self.model_folder, "training-simtrace")):
            self.type = LogFolderType.DRFC_MODEL_SINGLE_WORKERS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = os.path.join(
                    self.model_folder, "training-simtrace", "*-iteration.csv")
                self.training_simtrace_split = r'(.*)/training-simtrace/(.*)-iteration.csv'
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder, "evaluation-*", "evaluation-simtrace", "0-iteration.csv")
                self.evaluation_simtrace_split = \
                    r'.*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv'

            if self.model_metadata_path is None:
                self.model_metadata_path = os.path.join(
                    self.model_folder, "model", "model_metadata.json")

            if self.hyperparameters_path is None:
                self.hyperparameters_path = os.path.join(
                    self.model_folder, "ip", "hyperparameters.json")

        elif os.path.isdir(os.path.join(self.model_folder, "0")):
            self.type = LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = os.path.join(
                    self.model_folder, "**", "training-simtrace", "*-iteration.csv")
                self.training_simtrace_split = r'.*/(.)/training-simtrace/(.*)-iteration.csv'
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder, "evaluation-*", "evaluation-simtrace", "0-iteration.csv")
                self.evaluation_simtrace_split = \
                    r'.*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv'

            if self.model_metadata_path is None:
                self.model_metadata_path = os.path.join(
                    self.model_folder, "model", "model_metadata.json")

            if self.hyperparameters_path is None:
                self.hyperparameters_path = os.path.join(
                    self.model_folder, "ip", "hyperparameters.json")

        elif os.path.isdir(os.path.join(self.model_folder, "model")):
            self.type = LogFolderType.DRFC_MODEL_UPLOAD
            if self.training_simtrace_path is None:
                self.training_simtrace_path = None
                self.training_simtrace_split = None
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder, "evaluation-*", "evaluation-simtrace", "0-iteration.csv")
                self.evaluation_simtrace_split = \
                    r'.*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv'

            if self.model_metadata_path is None:
                self.model_metadata_path = os.path.join(
                    self.model_folder, "model", "model_metadata.json")

            if self.hyperparameters_path is None:
                self.hyperparameters_path = os.path.join(
                    self.model_folder, "ip", "hyperparameters.json")

        return self.type


class S3FileHandler(FileHandler):
    """An implementation of the FileHandler interface that interacts with an S3 bucket.

    Methods are exposed to detect what kind of model folder is present, to list files
    and to read files into a buffer.
    """

    def __init__(self, bucket: str, prefix: str = None,
                 s3_endpoint_url: str = None, region: str = None, profile: str = None,
                 ):
        """Initializes an S3 file handler.

        The bucket can either be in real S3, or in a locally hosted S3 using minio.

        Args:
            bucket:
                Name of the bucket
            prefix:
                Prefix pointing to the root of the model folder.
            s3_endpoint_url:
                Alternate Endpoint URL, used for locally hosted S3.
            region:
                AWS region to connect to. Not applicable for locally hosted S3.
            profile:
                Name of the profile in `.aws/` which contains the credentials to use.

        """
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

    def list_files(self, filterexp: str = None, check_exist: bool = False) -> list:
        """Lists a set of files.

        For S3 buckets the files in `s3://{bucket}/{prefix}` is listed, and subsequently
        filtered with the expression in `filterexp`. The resulting list is returned.
        If `filterexp = None` then the entire list is returned.

        Args:
            filterexp:
                Filter expression. Typically the values for `training_simtrace_path`,
                `evaluation_simtrace_path` etc. are provided.
            check_exist:
                If `True` then raise Exception if no files are found.
                Otherwise return empty list.

        Returns:
            A list of files.
        """
        files = []

        if check_exist and filterexp is None:
            raise Exception("File path is not defined.")

        bucket_obj = self.s3.Bucket(self.bucket)
        for objects in bucket_obj.objects.filter(Prefix=self.prefix):
            files.append(objects.key)

        if filterexp is not None:
            return_files = [x for x in files if re.match(filterexp, x)]
        else:
            return_files = files

        if len(return_files) > 0:
            return return_files
        else:
            if check_exist:
                raise Exception("No files found in s3://{}/{} using filter {}"
                                .format(self.bucket, self.prefix, filterexp))
            else:
                return []

    def get_file(self, key: str) -> bytes:
        """Downloads a given gile as byte array.

        Args:
            key:
                Path to a file within a bucket.

        Returns:
            A bytes object containing the file.
        """
        bytes_io = BytesIO()
        self.s3.Object(self.bucket, key).download_fileobj(bytes_io)
        return bytes_io.getvalue()

    def determine_root_folder_type(self) -> LogFolderType:

        if len(self.list_files(filterexp=(self.prefix + r'sim-trace/(.*)'))) > 0:
            self.type = LogFolderType.CONSOLE_MODEL_WITH_LOGS
            self.training_simtrace_path = self.prefix + \
                r'sim-trace/training/training-simtrace/(.*)-iteration\.csv'
            self.training_simtrace_split = \
                r'(.*)/sim-trace/training/training-simtrace/(.*)-iteration.csv'
            self.evaluation_simtrace_path = self.prefix + \
                r'sim-trace/evaluation/(.*)/evaluation-simtrace/0-iteration\.csv'
            self.evaluation_simtrace_split = \
                r'.*sim-trace/evaluation/([0-9]{14})-.*/evaluation-simtrace/(.*)-iteration\.csv'
            self.training_robomaker_log_path = self.prefix + \
                r'logs/training/(.*)-robomaker\.log'
            self.evaluation_robomaker_log_path = self.prefix + \
                r'logs/evaluation/(.*)-robomaker\.log'
            self.evaluation_robomaker_split = \
                r'.*/evaluation/evaluation-([0-9]{14})-(.*)-robomaker.log'
            self.leaderboard_robomaker_log_path = self.prefix + \
                r'logs/leaderboard/(.*)-robomaker\.log'
            self.leaderboard_robomaker_log_split = \
                r'.*/leaderboard/leaderboard-([0-9]{14})-(.*)-robomaker.log'
            self.model_metadata_path = self.prefix + \
                r'model/model_metadata.json'
            self.hyperparameters_path = self.prefix + \
                r'ip/hyperparameters.json'

        elif len(self.list_files(filterexp=(self.prefix + r'training-simtrace/(.*)'))) > 0:
            self.type = LogFolderType.DRFC_MODEL_SINGLE_WORKERS
            self.training_simtrace_path = self.prefix + r'training-simtrace/(.*)-iteration\.csv'
            self.training_simtrace_split = r'(.*)/training-simtrace/(.*)-iteration.csv'
            self.evaluation_simtrace_path = self.prefix + \
                r'evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv'
            self.evaluation_simtrace_split = \
                r'.*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv'
            self.model_metadata_path = self.prefix + \
                r'model/model_metadata.json'
            self.hyperparameters_path = self.prefix + \
                r'ip/hyperparameters.json'

        elif len(self.list_files(filterexp=(self.prefix + r'./training-simtrace/(.*)'))) > 0:
            self.type = LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS
            self.training_simtrace_path = self.prefix + \
                r'(.)/training-simtrace/(.*)-iteration\.csv'
            self.training_simtrace_split = r'.*/(.)/training-simtrace/(.*)-iteration.csv'
            self.evaluation_simtrace_path = self.prefix + \
                r'evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv'
            self.evaluation_simtrace_split = \
                r'.*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv'
            self.model_metadata_path = self.prefix + \
                r'model/model_metadata.json'
            self.hyperparameters_path = self.prefix + \
                r'ip/hyperparameters.json'

        elif len(self.list_files(filterexp=(self.prefix + r'evaluation-([0-9]{14})'))) > 0:
            self.type = LogFolderType.DRFC_MODEL_UPLOAD
            self.training_simtrace_path = None
            self.training_simtrace_split = None
            self.evaluation_simtrace_path = self.prefix + \
                r'evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv'
            self.evaluation_simtrace_split = \
                r'.*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv'
            self.model_metadata_path = self.prefix + \
                r'model/model_metadata.json'
            self.hyperparameters_path = self.prefix + \
                r'ip/hyperparameters.json'

        return self.type
