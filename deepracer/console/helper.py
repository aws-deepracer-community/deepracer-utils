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

from io import BytesIO, TextIOWrapper
from urllib.request import urlopen
import tarfile
import json

import boto3

import numpy as np
import pandas as pd

from botocore.exceptions import ClientError
from deepracer import boto3_enhancer
from deepracer.logs import SimulationLogsIO


class ConsoleHelper:
    def __init__(self, profile=None, region='us-east-1'):

        if profile is not None:
            session = boto3.session.Session(profile_name=profile)
        else:
            session = boto3.session.Session()
        self.dr = boto3_enhancer.deepracer_client(session=session, region_name=region)

    def find_model(self, model_name):

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

    def get_training_job(self, model_arn):
        m_response = self.dr.list_training_jobs(ModelArn=model_arn)
        m_jobs = m_response["TrainingJobs"]
        training_job_arn = m_jobs[0]['JobArn']
        m_response = self.dr.get_training_job(TrainingJobArn=training_job_arn)
        m_job = m_response['TrainingJob']
        return m_job

    def get_training_log_robomaker(self, model_arn, data=None):

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
