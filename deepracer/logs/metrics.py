"""
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

import json
import math
from io import BytesIO
from typing import TypeVar
from urllib.request import urlopen

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class TrainingMetrics:
    """ Class used to load in training metrics from S3
    """

    def __init__(
            self,
            bucket: str,
            model_name: str = None,
            pattern: str = "{}/metrics/TrainingMetrics{}.json",
            s3_endpoint_url: str = None,
            region: str = None,
            profile: str = None,
            fname: str = None,
            url: str = None,
            training_round: int = 1,
            display_digits_iteration: int = 3,
            display_digits_episode: int = 4,
            display_digits_round: int = 2
    ):
        """Creates a TrainingMetrics object. Loads the first metrics file into a DataFrame if
            model name is provided.

        Arguments:
        bucket - S3 bucket where the metrics file is stored
        model_name - (str) Name of the model that will be loaded. Default None - returns empty
            object.
        pattern - (str) Filename pattern that will be formatted with the model name to create
            the key.
        training_round - (int) Integer value that will be used to distinguish data.
        display_digits_iteration - (int) Number that will define the padding (e.g for round 1,
            iteration 25 the display_digits_iteration=3 would give unique index as 1-025)
        display_digits_episode - (int) Number that will define the padding (e.g for round 1,
            iteration 25 the display_digits_iteration=4 would give unique index as 1-1225)
        s3_endpoint_url - (str) URL for the S3 endpoint
        region - (str) AWS Region for S3
        profile - (str) Local awscli profile to use when connecting
        fname - (str) Read in a file, rather than from S3

        Returns:
        TrainingMetrics object.
        """
        if profile is not None:
            session = boto3.session.Session(profile_name=profile)
            self.s3 = session.resource("s3", endpoint_url=s3_endpoint_url, region_name=region)
        else:
            self.s3 = boto3.resource("s3", endpoint_url=s3_endpoint_url, region_name=region)
        self.max_iteration_strlen = display_digits_iteration
        self.max_episode_strlen = display_digits_episode
        self.max_round_strlen = display_digits_round
        self.metrics = None
        self.bucket = bucket
        self.pattern = pattern
        if model_name is not None:
            data = self._getS3File(bucket, self.pattern.format(model_name, ''), False)
            df = self._loadRound(
                data, training_round
            )
            self.metrics = df

        if fname is not None:
            data = self._getFile(fname, False)
            df = self._loadRound(
                data, training_round
            )
            self.metrics = df

        if url is not None:
            data = self._getUrlFile(url, False)
            df = self._loadRound(
                data, training_round
            )
            self.metrics = df

    def _getFile(self, fname: str, verbose=False) -> dict:
        if verbose:
            print("Reading in file://%s" % (fname))

        f = open(fname)
        data = json.load(f)
        f.close()

        return data

    def _getS3File(self, bucket: str, key: str, verbose=False) -> dict:

        if verbose:
            print("Downloading s3://%s/%s" % (bucket, key))

        bytes_io = BytesIO()
        self.s3.Object(bucket, key).download_fileobj(bytes_io)
        data = json.loads(bytes_io.getvalue())

        return data

    def _getUrlFile(self, url: str, verbose=False) -> dict:
        if verbose:
            print("Downloading %s" % (url))

        bytes_io = BytesIO(urlopen(url).read())
        data = json.loads(bytes_io.getvalue())

        return data

    def _loadRound(self, data: dict, training_round: int = 1, worker: int = 0,
                   verbose=False) -> pd.DataFrame:

        df = pd.DataFrame(data["metrics"])
        if worker == 0:
            self.episodes_per_iteration = max(df["trial"])

        df["round"] = training_round
        df["iteration"] = (
            ((df["episode"] - 1) / self.episodes_per_iteration)
            .apply(np.floor)
            .astype(int)
        )
        if self.metrics is not None:
            prev_metrics = self.metrics[self.metrics["round"] < training_round]
            if prev_metrics.shape[0] > 0:
                df["master_iteration"] = (
                    max(prev_metrics["master_iteration"]) + 1 + df["iteration"]
                )
            else:
                df["master_iteration"] = df["iteration"]
        else:
            df["master_iteration"] = df["iteration"]
        self.max_iteration_strlen = max(
            len(str(max(df["iteration"]))), self.max_iteration_strlen
        )
        df["r-i"] = (
            df["round"].astype(str).str.pad(width=self.max_round_strlen, side="left", fillchar="0")
            + "-"
            + df["iteration"]
            .astype(str)
            .str.pad(width=self.max_iteration_strlen, side="left", fillchar="0")
        )

        df["r-e"] = (
            df["round"].astype(str)
            + "-"
            + df["episode"]
            .astype(str)
            .str.pad(width=self.max_episode_strlen, side="left", fillchar="0")
        )
        df["worker"] = worker
        df["reward"] = df["reward_score"]
        df["completion"] = df["completion_percentage"]
        df["complete"] = df["episode_status"].apply(
            lambda x: 1 if x == "Lap complete" else 0
        )
        df["time"] = df["elapsed_time_in_milliseconds"] / 1000
        print(
            ("Successfully loaded training round %i for worker %i: Iterations: %i, " +
             "Training episodes: %i, Evaluation episodes: %i")
            % (
                training_round,
                worker,
                max(df["iteration"]) + 1,
                max(df["episode"]),
                df[df["phase"] == "evaluation"].shape[0],
            )
        )
        return df[
            [
                "r-i",
                "round",
                "iteration",
                "master_iteration",
                "episode",
                "r-e",
                "worker",
                "trial",
                "phase",
                "reward",
                "completion",
                "time",
                "complete",
                "start_time",
            ]
        ]

    def addRound(self, model_name: str, fname: str = None, training_round: int = 2,
                 workers: int = 1, worker: int = None):
        """Adds a round of training metrics to the data set

        Arguments:
        model_name - (str) Name of the model that will be loaded.
        fname - (str) Read from file not from S3
        training_round - (int) Integer value that will be used to distinguish data.
        workers - (int) Number of separate workers files to be loaded. (Default: 1)
        """

        if fname is not None:
            data = self._getFile(fname, False)

            if worker is not None:
                df = self._loadRound(
                    data, training_round, worker
                )
            else:
                df = self._loadRound(
                    data, training_round
                )

            if self.metrics is not None:
                self.metrics = pd.concat([self.metrics, df], ignore_index=True)
            else:
                self.metrics = df

        else:
            for w in range(0, workers):
                if w > 0:
                    worker_suffix = "_{}".format(w)
                else:
                    worker_suffix = ""

                data = self._getS3File(self.bucket, self.pattern.format(model_name, worker_suffix))
                df = self._loadRound(
                    data, training_round, w
                )

                if self.metrics is not None:
                    self.metrics = pd.concat([self.metrics, df], ignore_index=True)
                else:
                    self.metrics = df

    def reloadRound(self, model_name: str, fname: str = None, training_round: int = 2,
                    workers: int = 1):
        """Adds a round of training metrics to the data set

        Arguments:
        model_name - (str) Name of the model that will be loaded.
        training_round - (int) Integer value that will be used to distinguish data.
        workers - (int) Number of separate workers files to be loaded. (Default: 1)
        """

        print("Reloading round {}".format(training_round))
        self.metrics = self.metrics[self.metrics['round'] != training_round]

        if fname is not None:
            data = self._getFile(fname, False)
            df = self._loadRound(
                data, training_round
            )

            if self.metrics is not None:
                self.metrics = self.metrics.append(df, ignore_index=True)
            else:
                self.metrics = df
        else:
            for w in range(0, workers):
                if w > 0:
                    worker_suffix = "_{}".format(w)
                else:
                    worker_suffix = ""

                data = self._getS3File(self.bucket, self.pattern.format(model_name, worker_suffix))
                df = self._loadRound(
                    data, training_round, w
                )

                if self.metrics is not None:
                    self.metrics = self.metrics.append(df, ignore_index=True)
                else:
                    self.metrics = df

    def getEvaluation(self) -> pd.DataFrame:
        """Get the Evaluation part of the data.

        Returns:
        Pandas DataFrame containing all evaluation episodes.
        """
        return self.metrics[self.metrics["phase"] == "evaluation"]

    def getTraining(self) -> pd.DataFrame:
        """Get the Training part of the data.

        Returns:
        Pandas DataFrame containing all training episodes.
        """
        return self.metrics[self.metrics["phase"] == "training"]

    def getSummary(self, rounds: list = None, method: str = "mean",
                   summary_index: list = ["r-i", "iteration"], workers: list = None,
                   completedLapsOnly=False) -> pd.DataFrame:
        """Provides summary per iteration. Data for evaluation and training is separated.

        Arguments:
        rounds - (list) Array of round numbers to be plotted.
        method - (str) Statistical value to be calculated. Examples are 'mean', 'median',
            'min' & 'max'. Default: 'mean'.
        summary_index - (list) List of columns to be used as index of summary.
            Default ['r-i','iteration'].
        completedLapsOnly - (boolean) True will include only completed laps in summary.
        rounds - (list) List of rounds to include in the summary. Defaults to all rounds.
        workers - (list) List of workers to include in the summary. Defaults to all workers.

        Returns:
        Pandas DataFrame containing the summary table.
        """
        input_df = self.metrics

        if workers is not None:
            input_df = input_df[input_df["worker"].isin(workers)]

        if rounds is not None:
            input_df = input_df[input_df["round"].isin(rounds)]

        if completedLapsOnly:
            input_df = input_df[input_df["complete"] == 1]

        columns = summary_index + ["reward", "completion", "time", "complete"]
        training_input = input_df[input_df["phase"] == "training"][columns].copy()
        eval_input = input_df[input_df["phase"] == "evaluation"][columns].copy()

        training_gb = training_input.groupby(summary_index)
        training_agg = getattr(training_gb, method)()
        training_agg.columns = [
            "train_reward",
            "train_completion",
            "train_time",
            "train_completed",
        ]

        training_cnt = training_gb.count()
        training_agg['train_episodes'] = training_cnt['complete']

        eval_gb = eval_input.groupby(summary_index)
        eval_agg = getattr(eval_gb, method)()
        eval_agg.columns = [
            "eval_reward",
            "eval_completion",
            "eval_time",
            "eval_completed",
        ]

        eval_cnt = eval_gb.count()
        eval_agg['eval_episodes'] = eval_cnt['complete']

        return pd.concat([training_agg, eval_agg], axis=1, sort=False)

    ListStr = TypeVar('ListStr', list, str)

    def plotProgress(
            self,
            method: ListStr = "mean",
            rolling_average: int = 5,
            figsize: tuple = (12, 5),
            rounds: list = None,
            workers: list = None,
            series: list = [
                ("eval_completion", "Evaluation", "orange"),
                ("train_completion", "Training", "blue"),
            ],
            title: str = "Completion per Iteration ({})",
            xlabel: str = "Iteration",
            ylabel: str = "Percent complete ({})",
            completedLapsOnly: bool = False,
            grid: bool = False,
    ) -> Figure:
        """Plots training progress. Allows selection of multiple iterations.

        Arguments:
        method - (str / list) Statistical value to be calculated. Examples are 'mean', 'median',
            'min' & 'max'. Default: 'mean'.
        rolling_average - (int) Plotted line will be averaged with last number of x iterations.
            Default: 5.
        figsize - (tuple) Matplotlib figsize definition.
        series - (list) List of series to plot, contains tuples containing column in summary to
            plot, the legend title and color of plot. Default:
            [('eval_completion','Evaluation','orange'),('train_completion','Training','blue')]
        title - (string) The title of the diagram. Optional formatting placeholder '{}' for
            the method.
        xlabel - (string) The x label of the diagram.
        ylabel - (string) The y label of the diagram. Optional formatting placeholder '{}'
            for the method.
        completedLapsOnly - (boolean) Include only completed laps in the statistics.
        grid - (boolean) Adds a grid to the plot.
        rounds - (list) List of rounds to include in the summary. Defaults to all rounds.
        workers - (list) List of workers to include in the summary. Defaults to all workers.

        Returns:
        Matplotlib Figure containing the plot.
        """

        plot_methods = []
        if type(method) is not list:
            plot_methods.append(method)
        else:
            plot_methods = method

        fig, axarr_raw = plt.subplots(1, len(plot_methods), figsize=figsize, sharey=True)

        axarr = []
        if type(axarr_raw) is not np.ndarray:
            axarr.append(axarr_raw)
        else:
            axarr = axarr_raw

        for (m, ax) in zip(plot_methods, axarr):
            summary = self.getSummary(method=m, rounds=rounds, workers=workers,
                                      completedLapsOnly=completedLapsOnly)
            labels = max(math.floor(summary.shape[0] / (15 / len(plot_methods))), 1)
            x = []
            t = []
            for i, a in enumerate(summary.index):
                x.append(a[0])
                if i % labels == 0:
                    t.append(a[0])

            for s in series:
                ax.scatter(x, summary[s[0]], s=2, alpha=0.5, color=s[2])
                ax.plot(
                    x,
                    summary[s[0]].rolling(rolling_average, min_periods=1).mean().values,
                    label=s[1],
                    color=s[2],
                )
            ax.set_title(title.format(m))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel.format(m))
            ax.set_xticks(t)

            ax.legend(loc='upper left')

            self.metrics["iteration"].unique()
            if rounds is not None:
                unique_rounds = rounds[1:]
            else:
                unique_rounds = self.metrics["round"].unique()[1:]

            for r in unique_rounds:
                label = "{}-{}".format(
                    str(r).zfill(self.max_round_strlen),
                    "0".zfill(self.max_iteration_strlen)
                )
                ax.axvline(x=label, dashes=[0.25, 0.75], linewidth=0.5, color="black")

            if grid:
                ax.grid()

        plt.show()
        return fig
