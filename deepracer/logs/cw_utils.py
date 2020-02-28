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

import boto3
import sys
import dateutil.parser
import os


class CloudWatchLogs:
    '''
    Set of methods to fetch DeepRacer Simulation logs from Amazon CloudWatch.
    You can use it for SageMaker logs as well, just change group and prefix.

    Uses Boto3:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/logs.html
    '''

    RACE_SIMULATION_GROUP = '/aws/deepracer/leaderboard/SimulationJobs'
    TRAINING_SIMULATION_GROUP = '/aws/robomaker/SimulationJobs'
    SAGEMAKER_GROUP = '/aws/sagemaker/TrainingJobs'

    @staticmethod
    def get_log_events(
        log_group,
        stream_name=None,
        stream_prefix=None,
        start_time=None,
        end_time=None
    ):
        """
        Fetch the logs stream and yield its messages to caller.

        Arguments:
        log_group - which group to look in for the stream
        stream_name - name of the stream to download. Required if
            stream_prefix is None
        stream_prefix - prefix of the name of the stream to
            download. Required if stream_name is None
        start_time - start time for stream's messages
        end_time - end time for stream's messages

        Yields:
        Subsequent portions of a streams list to iterate over
        """
        client = boto3.client('logs')
        if stream_name is None and stream_prefix is None:
            print("both stream name and prefix can't be None")
            return

        kwargs = {
            'logGroupName': log_group,
            'logStreamNames': [stream_name],
            'limit': 10000,
        }

        if stream_prefix:
            kwargs = {
                'logGroupName': log_group,
                'logStreamNamePrefix': stream_prefix,
                'limit': 10000,
            }

        kwargs['startTime'] = start_time
        kwargs['endTime'] = end_time

        while True:
            resp = client.filter_log_events(**kwargs)
            yield from resp['events']
            try:
                kwargs['nextToken'] = resp['nextToken']
            except KeyError:
                break

    @staticmethod
    def download_log(fname, stream_name=None, stream_prefix=None,
                     log_group=None, start_time=None, end_time=None, force=False):
        """
        Downloads the content of a log stream to save it to a file. By default
        the method only downloads logs if a given file doesn't yet exist. It
        also downloads messages in the stream between 2016 and 2033 which is
        a very long period understood to save all messages in the stream.

        Arguments:
        fname - location and name of the file to save logs to
        stream_name - name of the stream to download. Required if
            stream_prefix is None
        stream_prefix - prefix of the name of the stream to
            download. Required if stream_name is None
        log_group - which group to look in for the stream
        start_time - start time, by default 1451490400000 which is
            December 30, 2015 3:46:40 PM UTC
        end_time - end time, by default 2000000000000 which is
            May 18, 2033 3:33:20 AM UTC
        force - if set to True, file specified in fname will be
            overwritten if it exists and logs will be downloaded
            again. Handy for when you download the stream of a live
            training; False by default
        """
        if os.path.isfile(fname) and not force:
            print('Log file exists, use force=True to download again')
            return

        if start_time is None:
            start_time = 1451490400000  # 2018
        if end_time is None:
            end_time = 2000000000000  # 2033 #arbitrary future date
        if log_group is None:
            log_group = "/aws/robomaker/SimulationJobs"

        with open(fname, 'w') as f:
            logs = CloudWatchLogs.get_log_events(
                log_group=log_group,
                stream_name=stream_name,
                stream_prefix=stream_prefix,
                start_time=start_time,
                end_time=end_time
            )
            for event in logs:
                f.write(event['message'].rstrip())
                f.write("\n")

    @staticmethod
    def download_all_logs(
        pathprefix,
        log_group,
        not_older_than=None,
        older_than=None,
        force=False
    ):
        """
        Download all log streams between dates.

        Arguments:
        pathprefix - prefix for a path into which to save the logs;
            pathprefix will be concatenated with a stream name prefix
            which is the part of the name from start to first "/"
            character if it exists or to the end if not; for value
            "abc/def" and stream "str123/asd..." the file path will be
            "abc/defstr123.log"; required
        log_group - which group to look in for the streams; required
        not_older_than - the oldest date at which a stream has to have
            at least one message to be downloaded; unlimited if not
            provided; ISO-8601 compliant date string. Example:
            "2020-02-20 02:02 UTC"
        older_than - the most recent date at which a stream has to have
            at least one message to be downloaded; unlimited if not
            provided; ISO-8601 compliant date string. Example:
            "2020-02-20 02:02 UTC"
        force - if set to True, files specified by pathprefix and
            stream name will be overwritten if they exist and logs
            will be downloaded again. Handy for when you download
            the stream of a live training; False by default

        Returns:
        List of fetched files in a tuple containing:
        - file location
        - stream name
        - timestamp of the first event in the stream
        - timestamp of the last event in the stream
        """
        client = boto3.client('logs')

        lower_timestamp = CloudWatchLogs.iso_to_timestamp(not_older_than)
        upper_timestamp = CloudWatchLogs.iso_to_timestamp(older_than)

        fetched_files = []
        next_token = None

        while next_token is not 'theEnd':
            streams = CloudWatchLogs.describe_log_streams(
                client, log_group, next_token)

            next_token = streams.get('nextToken', 'theEnd')

            for stream in streams['logStreams']:
                if lower_timestamp and stream['lastEventTimestamp'] < lower_timestamp:
                    return fetched_files  # we're done, next logs will be even older
                if upper_timestamp and stream['firstEventTimestamp'] > upper_timestamp:
                    continue
                stream_prefix = stream['logStreamName'].split("/")[0]
                file_name = "%s%s.log" % (pathprefix, stream_prefix)

                if not os.path.isfile(file_name) or force:
                    CloudWatchLogs.download_log(
                        file_name, stream_prefix=stream_prefix, log_group=log_group)

                fetched_files.append(
                    (
                        file_name,
                        stream_prefix,
                        stream['firstEventTimestamp'],
                        stream['lastEventTimestamp']
                    )
                )

        return fetched_files

    @staticmethod
    def describe_log_streams(client, log_group, next_token=None):
        """
        Fetch description of the log streams in a group.
        If next_token is supplied, include it in the call

        Arguments:
        client - Boto3 client to use to fetch the descriptions
        log_group - which group to look in for the streams; required
        next_token - token value to fetch the next portion of descriptions

        Returns:
        A list of streams descriptions
        """
        if next_token:
            streams = client.describe_log_streams(logGroupName=log_group, orderBy='LastEventTime',
                                                  descending=True, nextToken=next_token)
        else:
            streams = client.describe_log_streams(logGroupName=log_group, orderBy='LastEventTime',
                                                  descending=True)
        return streams

    @staticmethod
    def iso_to_timestamp(iso_date):
        """
        Convert date from ISO String to a timestamp

        Arguments:
        iso_date - ISO-8601 compliant date string. Example: "2020-02-20 02:02 UTC"

        Returns:
        A timestamp in miliseconds for a given iso_date or None if iso_date is None
        """
        return dateutil.parser.parse(iso_date).timestamp() * 1000 if iso_date else None
