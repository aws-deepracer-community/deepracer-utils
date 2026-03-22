import glob
import os
import re
import tarfile
import threading
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

        # Normalise to forward slashes so the split regexes work on all platforms.
        return_files = [p.replace("\\", "/") for p in return_files]

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
        with open(key, "rb") as fh:
            bytes_io = BytesIO(fh.read())
        return bytes_io.getvalue()

    def determine_root_folder_type(self) -> LogFolderType:

        if os.path.isdir(os.path.join(self.model_folder, "sim-trace")):
            if os.path.isdir(
                os.path.join(self.model_folder, "sim-trace", "training", "training-simtrace")
            ):
                # Old console format: training-simtrace sits directly under sim-trace/training/
                self.type = LogFolderType.CONSOLE_MODEL_WITH_LOGS
                if self.training_simtrace_path is None:
                    self.training_simtrace_path = os.path.join(
                        self.model_folder,
                        "sim-trace",
                        "training",
                        "training-simtrace",
                        "*-iteration.csv",
                    )
                    self.training_simtrace_split = (
                        r"(.*)/training/training-simtrace/(.*)-iteration.csv"
                    )
                    self.evaluation_simtrace_path = os.path.join(
                        self.model_folder,
                        "sim-trace",
                        "evaluation",
                        "*",
                        "evaluation-simtrace",
                        "0-iteration.csv",
                    )
                    self.evaluation_simtrace_split = (
                        r".*/evaluation/([0-9]{14})-.*/evaluation-simtrace/(.*)-iteration\.csv"
                    )
                if self.training_robomaker_log_path is None:
                    self.training_robomaker_log_path = glob.glob(
                        os.path.join(self.model_folder, "**", "training", "*-robomaker.log")
                    )[0]

                if self.evaluation_robomaker_log_path is None:
                    self.evaluation_robomaker_log_path = os.path.join(
                        self.model_folder, "**", "evaluation", "*-robomaker.log"
                    )

                if self.evaluation_robomaker_split is None:
                    self.evaluation_robomaker_split = (
                        r".*/evaluation/evaluation-([0-9]{14})-(.*)-robomaker.log"
                    )

                if self.leaderboard_robomaker_log_path is None:
                    self.leaderboard_robomaker_log_path = os.path.join(
                        self.model_folder, "**", "leaderboard", "*-robomaker.log"
                    )

                if self.leaderboard_robomaker_log_split is None:
                    self.leaderboard_robomaker_log_split = (
                        r".*/leaderboard/leaderboard-([0-9]{14})-(.*)-robomaker.log"
                    )

                if self.model_metadata_path is None:
                    self.model_metadata_path = os.path.join(
                        self.model_folder, "model", "model_metadata.json"
                    )

                if self.hyperparameters_path is None:
                    self.hyperparameters_path = os.path.join(
                        self.model_folder, "ip", "hyperparameters.json"
                    )
            else:
                # New console format (v2): extra ISO-8601 timestamp subdirectory under
                # sim-trace/training/ before training-simtrace/
                self.type = LogFolderType.DROA_SOLUTION_LOGS
                if self.training_simtrace_path is None:
                    self.training_simtrace_path = os.path.join(
                        self.model_folder,
                        "sim-trace",
                        "training",
                        "*",
                        "training-simtrace",
                        "*-iteration.csv",
                    )
                    self.training_simtrace_split = (
                        r"(.*)/training/[^/]+/training-simtrace/(.*)-iteration\.csv"
                    )
                    self.evaluation_simtrace_path = os.path.join(
                        self.model_folder,
                        "sim-trace",
                        "evaluation",
                        "*",
                        "evaluation-simtrace",
                        "*-iteration.csv",
                    )
                    self.evaluation_simtrace_split = (
                        r".*/evaluation/([^/]+)/evaluation-simtrace/(.*)-iteration\.csv"
                    )
                if self.training_robomaker_log_path is None:
                    candidates = glob.glob(
                        os.path.join(self.model_folder, "logs", "training", "*-simulation.log")
                    )
                    self.training_robomaker_log_path = candidates[0] if candidates else None

                if self.evaluation_robomaker_log_path is None:
                    self.evaluation_robomaker_log_path = os.path.join(
                        self.model_folder, "logs", "evaluation", "*-simulation.log"
                    )

                if self.evaluation_robomaker_split is None:
                    self.evaluation_robomaker_split = r".*/evaluation/([^/]+)-simulation\.log"

                if self.leaderboard_robomaker_log_path is None:
                    self.leaderboard_robomaker_log_path = os.path.join(
                        self.model_folder, "logs", "leaderboard", "*-simulation.log"
                    )

                if self.leaderboard_robomaker_log_split is None:
                    self.leaderboard_robomaker_log_split = r".*/leaderboard/([^/]+)-simulation\.log"

                if self.model_metadata_path is None:
                    self.model_metadata_path = os.path.join(
                        self.model_folder, "model", "model_metadata.json"
                    )

                if self.hyperparameters_path is None:
                    self.hyperparameters_path = os.path.join(
                        self.model_folder, "ip", "hyperparameters.json"
                    )

        elif os.path.isdir(os.path.join(self.model_folder, "training-simtrace")):
            self.type = LogFolderType.DRFC_MODEL_SINGLE_WORKERS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = os.path.join(
                    self.model_folder, "training-simtrace", "*-iteration.csv"
                )
                self.training_simtrace_split = r"(.*)/training-simtrace/(.*)-iteration.csv"
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder, "evaluation-*", "evaluation-simtrace", "0-iteration.csv"
                )
                self.evaluation_simtrace_split = (
                    r".*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
                )

            if self.model_metadata_path is None:
                self.model_metadata_path = os.path.join(
                    self.model_folder, "model", "model_metadata.json"
                )

            if self.hyperparameters_path is None:
                self.hyperparameters_path = os.path.join(
                    self.model_folder, "ip", "hyperparameters.json"
                )

        elif os.path.isdir(os.path.join(self.model_folder, "0")):
            self.type = LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS
            if self.training_simtrace_path is None:
                self.training_simtrace_path = os.path.join(
                    self.model_folder, "**", "training-simtrace", "*-iteration.csv"
                )
                self.training_simtrace_split = r".*/(.)/training-simtrace/(.*)-iteration.csv"
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder, "evaluation-*", "evaluation-simtrace", "0-iteration.csv"
                )
                self.evaluation_simtrace_split = (
                    r".*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
                )

            if self.model_metadata_path is None:
                self.model_metadata_path = os.path.join(
                    self.model_folder, "model", "model_metadata.json"
                )

            if self.hyperparameters_path is None:
                self.hyperparameters_path = os.path.join(
                    self.model_folder, "ip", "hyperparameters.json"
                )

        elif os.path.isdir(os.path.join(self.model_folder, "model")):
            self.type = LogFolderType.DRFC_MODEL_UPLOAD
            if self.training_simtrace_path is None:
                self.training_simtrace_path = None
                self.training_simtrace_split = None
                self.evaluation_simtrace_path = os.path.join(
                    self.model_folder, "evaluation-*", "evaluation-simtrace", "0-iteration.csv"
                )
                self.evaluation_simtrace_split = (
                    r".*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
                )

            if self.model_metadata_path is None:
                self.model_metadata_path = os.path.join(
                    self.model_folder, "model", "model_metadata.json"
                )

            if self.hyperparameters_path is None:
                self.hyperparameters_path = os.path.join(
                    self.model_folder, "ip", "hyperparameters.json"
                )

        return self.type


class S3FileHandler(FileHandler):
    """An implementation of the FileHandler interface that interacts with an S3 bucket.

    Methods are exposed to detect what kind of model folder is present, to list files
    and to read files into a buffer.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = None,
        s3_endpoint_url: str = None,
        region: str = None,
        profile: str = None,
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

        if prefix[-1:] == "/":
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
                raise Exception(
                    "No files found in s3://{}/{} using filter {}".format(
                        self.bucket, self.prefix, filterexp
                    )
                )
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

        if (
            len(self.list_files(filterexp=(self.prefix + r"sim-trace/training/training-simtrace/")))
            > 0
        ):
            # Old console format: training-simtrace sits directly under sim-trace/training/
            self.type = LogFolderType.CONSOLE_MODEL_WITH_LOGS
            self.training_simtrace_path = (
                self.prefix + r"sim-trace/training/training-simtrace/(.*)-iteration\.csv"
            )
            self.training_simtrace_split = (
                r"(.*)/sim-trace/training/training-simtrace/(.*)-iteration.csv"
            )
            self.evaluation_simtrace_path = (
                self.prefix + r"sim-trace/evaluation/(.*)/evaluation-simtrace/0-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*sim-trace/evaluation/([0-9]{14})-.*/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.training_robomaker_log_path = self.prefix + r"logs/training/(.*)-robomaker\.log"
            self.evaluation_robomaker_log_path = (
                self.prefix + r"logs/evaluation/(.*)-robomaker\.log"
            )
            self.evaluation_robomaker_split = (
                r".*/evaluation/evaluation-([0-9]{14})-(.*)-robomaker.log"
            )
            self.leaderboard_robomaker_log_path = (
                self.prefix + r"logs/leaderboard/(.*)-robomaker\.log"
            )
            self.leaderboard_robomaker_log_split = (
                r".*/leaderboard/leaderboard-([0-9]{14})-(.*)-robomaker.log"
            )
            self.model_metadata_path = self.prefix + r"model/model_metadata.json"
            self.hyperparameters_path = self.prefix + r"ip/hyperparameters.json"

        elif len(self.list_files(filterexp=(self.prefix + r"sim-trace/training/"))) > 0:
            # New console format (v2): extra ISO-8601 timestamp subdirectory under
            # sim-trace/training/ before training-simtrace/
            self.type = LogFolderType.DROA_SOLUTION_LOGS
            self.training_simtrace_path = (
                self.prefix + r"sim-trace/training/[^/]+/training-simtrace/(.*)-iteration\.csv"
            )
            self.training_simtrace_split = (
                r"(.*)/sim-trace/training/[^/]+/training-simtrace/(.*)-iteration\.csv"
            )
            self.evaluation_simtrace_path = (
                self.prefix + r"sim-trace/evaluation/[^/]+/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*/sim-trace/evaluation/([^/]+)/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.training_robomaker_log_path = self.prefix + r"logs/training/(.*)-simulation\.log"
            self.evaluation_robomaker_log_path = (
                self.prefix + r"logs/evaluation/(.*)-simulation\.log"
            )
            self.evaluation_robomaker_split = r".*/evaluation/([^/]+)-simulation\.log"
            self.leaderboard_robomaker_log_path = (
                self.prefix + r"logs/leaderboard/(.*)-simulation\.log"
            )
            self.leaderboard_robomaker_log_split = r".*/leaderboard/([^/]+)-simulation\.log"
            self.model_metadata_path = self.prefix + r"model/model_metadata.json"
            self.hyperparameters_path = self.prefix + r"ip/hyperparameters.json"

        elif len(self.list_files(filterexp=(self.prefix + r"sim-trace/evaluation/"))) > 0:
            # New console format (v2): evaluation-only archive
            self.type = LogFolderType.DROA_SOLUTION_LOGS
            self.training_simtrace_path = None
            self.training_simtrace_split = None
            self.evaluation_simtrace_path = (
                self.prefix + r"sim-trace/evaluation/[^/]+/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*/sim-trace/evaluation/([^/]+)/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.evaluation_robomaker_log_path = (
                self.prefix + r"logs/evaluation/(.*)-simulation\.log"
            )
            self.evaluation_robomaker_split = r".*/evaluation/([^/]+)-simulation\.log"
            self.leaderboard_robomaker_log_path = (
                self.prefix + r"logs/leaderboard/(.*)-simulation\.log"
            )
            self.leaderboard_robomaker_log_split = r".*/leaderboard/([^/]+)-simulation\.log"
            self.model_metadata_path = self.prefix + r"model/model_metadata.json"
            self.hyperparameters_path = self.prefix + r"ip/hyperparameters.json"

        elif len(self.list_files(filterexp=(self.prefix + r"training-simtrace/(.*)"))) > 0:
            self.type = LogFolderType.DRFC_MODEL_SINGLE_WORKERS
            self.training_simtrace_path = self.prefix + r"training-simtrace/(.*)-iteration\.csv"
            self.training_simtrace_split = r"(.*)/training-simtrace/(.*)-iteration.csv"
            self.evaluation_simtrace_path = (
                self.prefix + r"evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.model_metadata_path = self.prefix + r"model/model_metadata.json"
            self.hyperparameters_path = self.prefix + r"ip/hyperparameters.json"

        elif len(self.list_files(filterexp=(self.prefix + r"./training-simtrace/(.*)"))) > 0:
            self.type = LogFolderType.DRFC_MODEL_MULTIPLE_WORKERS
            self.training_simtrace_path = self.prefix + r"(.)/training-simtrace/(.*)-iteration\.csv"
            self.training_simtrace_split = r".*/(.)/training-simtrace/(.*)-iteration.csv"
            self.evaluation_simtrace_path = (
                self.prefix + r"evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.model_metadata_path = self.prefix + r"model/model_metadata.json"
            self.hyperparameters_path = self.prefix + r"ip/hyperparameters.json"

        elif len(self.list_files(filterexp=(self.prefix + r"evaluation-([0-9]{14})"))) > 0:
            self.type = LogFolderType.DRFC_MODEL_UPLOAD
            self.training_simtrace_path = None
            self.training_simtrace_split = None
            self.evaluation_simtrace_path = (
                self.prefix + r"evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.model_metadata_path = self.prefix + r"model/model_metadata.json"
            self.hyperparameters_path = self.prefix + r"ip/hyperparameters.json"

        return self.type


class TarFileHandler(FileHandler):
    """An implementation of the FileHandler interface that reads directly from a .tar.gz archive.

    Allows users to point at a downloaded console log archive without extracting it first::

        log = DeepRacerLog(TarFileHandler("deepracerindy-training-ACGVRmRuFNU9NkQ-logs.tar.gz"))

    The archive member paths are exposed to callers with the top-level wrapper directory
    stripped, so they are relative to the model root (e.g. ``sim-trace/training/…``).
    """

    def __init__(self, archive_path: str):
        """Opens the .tar.gz archive and determines the top-level prefix to strip.

        Args:
            archive_path:
                Path to the .tar.gz file on the local filesystem.
        """
        self.archive_path = archive_path
        self._tar = tarfile.open(archive_path, "r:gz")
        self._lock = threading.Lock()

        # Detect a single top-level wrapper directory (e.g. "ACGVRmRuFNU9NkQ/")
        # and record it so we can strip it from all member paths.
        top_dirs = {m.name.split("/")[0] for m in self._tar.getmembers() if "/" in m.name}
        self._prefix = (top_dirs.pop() + "/") if len(top_dirs) == 1 else ""

    def __del__(self):
        if hasattr(self, "_tar") and self._tar:
            self._tar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._tar:
            self._tar.close()

    def list_files(self, filterexp: str = None, check_exist: bool = False) -> list:
        """Lists files in the archive with the top-level prefix stripped.

        Args:
            filterexp:
                Optional regex pattern matched against the stripped member paths.
            check_exist:
                If ``True`` raise an exception when no files match.

        Returns:
            A list of stripped member paths (relative to the model root).
        """
        if check_exist and filterexp is None:
            raise Exception("File path is not defined.")

        all_files = [
            m.name[len(self._prefix) :]
            for m in self._tar.getmembers()
            if m.isfile() and m.name.startswith(self._prefix)
        ]

        if filterexp is not None:
            return_files = [p for p in all_files if re.search(filterexp, p)]
        else:
            return_files = all_files

        if return_files:
            return return_files
        if check_exist:
            raise Exception(
                "No files found in {} using filter {}".format(self.archive_path, filterexp)
            )
        return []

    def get_file(self, key: str) -> bytes:
        """Reads a single archive member into bytes.

        Args:
            key:
                Stripped member path as returned by :meth:`list_files`.

        Returns:
            A bytes object containing the file contents.
        """
        full_key = self._prefix + key
        with self._lock:
            f = self._tar.extractfile(self._tar.getmember(full_key))
            return f.read()

    def determine_root_folder_type(self) -> LogFolderType:

        if self.list_files(
            filterexp=r"sim-trace/training/[^/]+/training-simtrace/.*-iteration\.csv"
        ):
            # New console format (v2): extra ISO-8601 timestamp subdirectory under
            # sim-trace/training/ before training-simtrace/
            self.type = LogFolderType.DROA_SOLUTION_LOGS
            self.training_simtrace_path = (
                r"sim-trace/training/[^/]+/training-simtrace/[^/]+-iteration\.csv"
            )
            self.training_simtrace_split = (
                r"(.*)/training/[^/]+/training-simtrace/(.*)-iteration\.csv"
            )
            self.evaluation_simtrace_path = (
                r"sim-trace/evaluation/[^/]+/evaluation-simtrace/[^/]+-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*/evaluation/([^/]+)/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.training_robomaker_log_path = r"logs/training/[^/]+-simulation\.log"
            self.evaluation_robomaker_log_path = r"logs/evaluation/[^/]+-simulation\.log"
            self.evaluation_robomaker_split = r"logs/evaluation/([^/]+)-simulation\.log"
            self.leaderboard_robomaker_log_path = r"logs/leaderboard/[^/]+-simulation\.log"
            self.leaderboard_robomaker_log_split = r"logs/leaderboard/([^/]+)-simulation\.log"
            self.model_metadata_path = None
            self.hyperparameters_path = None

        elif self.list_files(
            filterexp=r"sim-trace/evaluation/[^/]+/evaluation-simtrace/.*-iteration\.csv"
        ):
            # New console format (v2): evaluation-only archive
            self.type = LogFolderType.DROA_SOLUTION_LOGS
            self.training_simtrace_path = None
            self.training_simtrace_split = None
            self.evaluation_simtrace_path = (
                r"sim-trace/evaluation/[^/]+/evaluation-simtrace/[^/]+-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*/evaluation/([^/]+)/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.evaluation_robomaker_log_path = r"logs/evaluation/[^/]+-simulation\.log"
            self.evaluation_robomaker_split = r"logs/evaluation/([^/]+)-simulation\.log"
            self.leaderboard_robomaker_log_path = r"logs/leaderboard/[^/]+-simulation\.log"
            self.leaderboard_robomaker_log_split = r"logs/leaderboard/([^/]+)-simulation\.log"
            self.model_metadata_path = None
            self.hyperparameters_path = None

        elif self.list_files(filterexp=r"sim-trace/training/training-simtrace/.*-iteration\.csv"):
            # Old console format: training-simtrace sits directly under sim-trace/training/
            self.type = LogFolderType.CONSOLE_MODEL_WITH_LOGS
            self.training_simtrace_path = (
                r"sim-trace/training/training-simtrace/[^/]+-iteration\.csv"
            )
            self.training_simtrace_split = r"(.*)/training/training-simtrace/(.*)-iteration\.csv"
            self.evaluation_simtrace_path = (
                r"sim-trace/evaluation/[^/]+/evaluation-simtrace/[^/]+-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*/evaluation/([0-9]{14})-.*/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.training_robomaker_log_path = r"logs/training/[^/]+-robomaker\.log"
            self.evaluation_robomaker_log_path = r"logs/evaluation/[^/]+-robomaker\.log"
            self.evaluation_robomaker_split = (
                r".*/evaluation/evaluation-([0-9]{14})-(.*)-robomaker\.log"
            )
            self.leaderboard_robomaker_log_path = r"logs/leaderboard/[^/]+-robomaker\.log"
            self.leaderboard_robomaker_log_split = (
                r".*/leaderboard/leaderboard-([0-9]{14})-(.*)-robomaker\.log"
            )
            self.model_metadata_path = None
            self.hyperparameters_path = None

        elif self.list_files(filterexp=r"training-simtrace/.*-iteration\.csv"):
            self.type = LogFolderType.DRFC_MODEL_SINGLE_WORKERS
            self.training_simtrace_path = r"training-simtrace/[^/]+-iteration\.csv"
            self.training_simtrace_split = r"(.*)/training-simtrace/(.*)-iteration\.csv"
            self.evaluation_simtrace_path = (
                r"evaluation-[0-9]{14}/evaluation-simtrace/[^/]+-iteration\.csv"
            )
            self.evaluation_simtrace_split = (
                r".*/evaluation-([0-9]{14})/evaluation-simtrace/(.*)-iteration\.csv"
            )
            self.model_metadata_path = r"model/model_metadata.json"
            self.hyperparameters_path = r"ip/hyperparameters.json"

        return self.type
