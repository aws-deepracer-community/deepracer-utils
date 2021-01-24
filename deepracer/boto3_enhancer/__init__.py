"""
Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Copyright 2020 AWS DeepRacer Community. All Rights Reserved.

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

import os
import boto3
import shutil

from deepracer.utils import DEEPRACER_UTILS_ROOT


DR_MODELS_ROOT = os.path.join(DEEPRACER_UTILS_ROOT, 'boto3_enhancer', 'models', "deepracer")


AWS_CLI_MODELS_DR_ROOT = os.path.join(os.path.expanduser("~"), ".aws", "models", "deepracer")


def add_deepracer(session=None, **kwargs):
    """
    Add deepracer service definition file to boto3 session.

    If session not provided, boto3.DEFAULT_SESSION is used.

    If boto3.DEFAULT_SESSION is not present, it is set up. kwargs are used then.

    We could have used AWS_DATA_PATH environment variable but I wanted to leave
    it untouched. Instead this is adding directly into the loader in session.

    This code has been written thanks to guidance of Don Barber of AWS. The model
    file for deepracer is his doing and has been introduced in
    """
    if not session:
        if not boto3.DEFAULT_SESSION:
            boto3.setup_default_session(**kwargs)
        session = boto3.DEFAULT_SESSION

    if DR_MODELS_ROOT not in session._loader.search_paths:
        session._loader.search_paths.append(DR_MODELS_ROOT)


def deepracer_client(region_name='us-east-1'):
    """
    Return deepracer client for boto3 with default (and only working) parameters
    """
    add_deepracer()

    return boto3.client('deepracer', region_name=region_name)


def install_deepracer_cli(force=False):
    if force:
        remove_deepracer_cli()

    if os.path.isdir(AWS_CLI_MODELS_DR_ROOT):
        print("Configuration already found, not installing. Use --force to overwrite")
        return

    shutil.copytree(DR_MODELS_ROOT, AWS_CLI_MODELS_DR_ROOT)


def remove_deepracer_cli():
    if not os.path.isdir(AWS_CLI_MODELS_DR_ROOT):
        print("Configuration doesn't exist, nothing to clean up")
        return

    shutil.rmtree(AWS_CLI_MODELS_DR_ROOT)
