from enum import Enum


class LogFolderType(Enum):
    CONSOLE_MODEL_WITH_LOGS = 0
    DRFC_MODEL_SINGLE_WORKERS = 1
    DRFC_MODEL_MULTIPLE_WORKERS = 2
    DRFC_MODEL_UPLOAD = 4
    UNKNOWN_FOLDER = 3


class LogType(Enum):
    TRAINING = 0
    EVALUATION = 1
    LEADERBOARD = 2
    NOT_DEFINED = 3
