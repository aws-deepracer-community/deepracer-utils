"""
Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Copyright 2019-2020 AWS DeepRacer Community. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import tarfile
from enum import Enum
from io import BytesIO, TextIOWrapper
from urllib.request import urlopen

import boto3
import pandas as pd

from deepracer import boto3_enhancer
from deepracer.logs import SimulationLogsIO


class LeaderboardSubmissionType(Enum):
    RANKED = 1
    LATEST = 2


class ConsoleHelper:
    """ Class used to retrieve information from AWS DeepRacer Console.
    """
    def __init__(self, profile: str = None, region: str = 'us-east-1'):

        if profile is not None:
            session = boto3.session.Session(profile_name=profile)
        else:
            session = boto3.session.Session()
        self.dr = boto3_enhancer.deepracer_client(session=session, region_name=region)

    def find_model(self, model_name: str) -> str:
        """Finds the Model ARN based on a model display name

        Arguments:
        model_name - (str) String with the model name

        Returns:
        String with the model ARN. None if not found.
        """

        m_response = self.dr.list_models(ModelType="REINFORCEMENT_LEARNING", MaxResults=25)
        model_dict = m_response["Models"]
        models = pd.DataFrame.from_dict(model_dict)
        my_model = models[models["ModelName"] == model_name]

        if my_model.size > 0:
            return my_model.loc[:, 'ModelArn'].values[0]

        while "NextToken" in m_response:
            m_response = self.dr.list_models(
                ModelType="REINFORCEMENT_LEARNING",
                MaxResults=50,
                NextToken=m_response["NextToken"],
            )
            model_dict = m_response["Models"]

            models = pd.DataFrame.from_dict(model_dict)
            my_model = models[models["ModelName"] == model_name]
            if my_model.size > 0:
                return my_model.loc[:, 'ModelArn'].values[0]

        return None

    def find_leaderboard(self, leaderboard_guid: str) -> str:
        """Finds the Leaderboard ARN based on a short GUID

        Arguments:
        leaderboard_guid - (str) String with the leaderboard GUID
                Example: 8fab8c08-66cd-4170-a0b0-ab2b464e42d9

        Returns:
        String with the model ARN. None if not found.
        """

        leaderboard_arn = "arn:aws:deepracer:::leaderboard/{}".format(leaderboard_guid)

        l_response = self.dr.list_leaderboards(MaxResults=25)
        lboards_dict = l_response["Leaderboards"]
        leaderboards = pd.DataFrame.from_dict(l_response["Leaderboards"])
        if leaderboards[leaderboards["Arn"] == leaderboard_arn].size > 0:
            return leaderboard_arn

        while "NextToken" in l_response:
            l_response = self.dr.list_leaderboards(
                MaxResults=50, NextToken=l_response["NextToken"]
            )
            lboards_dict = l_response["Leaderboards"]

            leaderboards = pd.DataFrame.from_dict(lboards_dict)
            if leaderboards[leaderboards["Arn"] == leaderboard_arn].size > 0:
                return leaderboard_arn

        return None

    def get_training_job(self, model_arn: str) -> dict:
        """Retrieves Training Job information for a Model

        Arguments:
        model_arn - (str) String with the Model ARN

        Returns:
        Dict with the Training Job information.
        """

        m_response = self.dr.list_training_jobs(ModelArn=model_arn)
        m_jobs = m_response["TrainingJobs"]
        training_job_arn = m_jobs[0]['JobArn']
        m_response = self.dr.get_training_job(TrainingJobArn=training_job_arn)
        m_job = m_response['TrainingJob']
        return m_job

    def get_training_log_robomaker(self, model_arn: str, data: list = None) -> pd.DataFrame:
        """Retrieves the Robomaker training log for a completed training

        Arguments:
        model_arn - (str) String with the Model ARN
        data - (list) Optional pass in a pre-filled data object

        Returns:
        Pandas DataFrame with the parsed log.
        """
        if data is None:
            data = []

        training_job = self.get_training_job(model_arn)
        training_job_arn = training_job['JobArn']
        f_url = self.dr.get_asset_url(Arn=training_job_arn, AssetType="LOGS")["Url"]
        bytes_io = BytesIO(urlopen(f_url).read())

        with tarfile.open(fileobj=bytes_io, mode="r:gz") as tar_file:
            for member in tar_file.getmembers():
                if member.name.find("robomaker") > 0:
                    log_buf = TextIOWrapper(tar_file.extractfile(member))
                    data = SimulationLogsIO.load_buffer(log_buf)
                    df = SimulationLogsIO.convert_to_pandas(data)
                    return df

        return None

    def get_leaderboard_log_robomaker(self, leaderboard_guid: str,
                                      select=LeaderboardSubmissionType.RANKED,
                                      data: list = None) -> pd.DataFrame:
        """Retrieves the Robomaker log for leaderboard submission

        Arguments:
        leaderboard_guid - (str) String with the Leaderboard GUID
        select - (LeaderboardSubmissionType) Get ranked or latest submission
        data - (list) Optional pass in a pre-filled data object

        Returns:
        Pandas DataFrame with the parsed log.
        """
        if data is None:
            data = []

        leaderboard_arn = self.find_leaderboard(leaderboard_guid)

        if select == LeaderboardSubmissionType.RANKED:
            response_m = self.dr.get_ranked_user_submission(LeaderboardArn=leaderboard_arn)
            submission = response_m["LeaderboardSubmission"]
            activity_arn = "arn:aws:deepracer:us-east-1:180406016328:leaderboard_evaluation_job/{}"\
                .format(submission["SubmissionVideoS3path"].split("/")[7].split("-", 2)[2])

        if leaderboard_arn is None:
            return None

        else:
            response_m = self.dr.get_latest_user_submission(LeaderboardArn=leaderboard_arn)
            submission = response_m["LeaderboardSubmission"]
            activity_arn = submission["ActivityArn"]

        if submission["LeaderboardSubmissionStatusType"] == "SUCCESS":
            f_url = self.dr.get_asset_url(Arn=activity_arn, AssetType="LOGS")["Url"]
            bytes_io = BytesIO(urlopen(f_url).read())

            with tarfile.open(fileobj=bytes_io, mode="r:gz") as tar_file:
                for member in tar_file.getmembers():
                    if member.name.find("robomaker") > 0:
                        log_buf = TextIOWrapper(tar_file.extractfile(member))
                        data = SimulationLogsIO.load_buffer(log_buf)
                        df = SimulationLogsIO.convert_to_pandas(data)
                        df['stream'] = member.name.split("/")[-1].split("-")[0]
                        return df
        else:
            return None
