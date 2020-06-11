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

from datetime import datetime
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from shapely.geometry.polygon import LineString

from .cw_utils import CloudWatchLogs as cw
from ..tracks.track_utils import Track


class SimulationLogsIO:
    """ Utilities for loading the logs
    """

    @staticmethod
    def load_single_file(fname, data=None):
        """Loads a single log file and remembers only the SIM_TRACE_LOG lines

        Arguments:
        fname - path to the file
        data - list to populate with SIM_TRACE_LOG lines. Default: None

        Returns:
        List of loaded log lines. If data is not None, it is the reference returned
        and the list referenced has new log lines appended
        """
        if data is None:
            data = []

        with open(fname, 'r') as f:
            for line in f.readlines():
                if "SIM_TRACE_LOG" in line:
                    parts = line.split("SIM_TRACE_LOG:")[1].split('\t')[0].split(",")
                    data.append(",".join(parts))

        return data

    @staticmethod
    def load_data(fname):
        """Load all log files for a given simulation

        Looks for all files for a given simulation and loads them. Takes the local training
        into account where in some cases the logs are split when they reach a certain size,
        and given a suffix .1, .2 etc.

        Arguments:
        fname - path to the file

        Returns:
        List of loaded log lines
        """
        from os.path import isfile
        data = []

        i = 1

        while isfile('%s.%s' % (fname, i)):
            SimulationLogsIO.load_single_file('%s.%s' % (fname, i), data)
            i += 1

        SimulationLogsIO.load_single_file(fname, data)

        if i > 1:
            print("Loaded %s log files (logs rolled over)" % i)

        return data

    @staticmethod
    def convert_to_pandas(data, episodes_per_iteration=20):
        """Load the log data to pandas dataframe

        Reads the loaded log files and parses them according to this format of print:

        stdout_ = 'SIM_TRACE_LOG:%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f,%s,%s,%.4f,%d,%.2f,%s\n' % (
                self.episodes, self.steps, model_location[0], model_location[1], model_heading,
                self.steering_angle,
                self.speed,
                self.action_taken,
                self.reward,
                self.done,
                all_wheels_on_track,
                current_progress,
                closest_waypoint_index,
                self.track_length,
                time.time())
            print(stdout_)

        Currently only supports 2019 logs but is forwards compatible.

        Arguments:
        data - list of log lines to parse
        episodes_per_iteration - value of the hyperparameter for a given training

        Returns:
        A pandas dataframe with loaded data
        """

        df_list = list()

        # ignore the first two dummy values that coach throws at the start.
        for d in data[2:]:
            parts = d.rstrip().split(",")
            episode = int(parts[0])
            steps = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            yaw = float(parts[4])
            steer = float(parts[5])
            throttle = float(parts[6])
            action = float(parts[7])
            reward = float(parts[8])
            done = 0 if 'False' in parts[9] else 1
            all_wheels_on_track = parts[10]
            progress = float(parts[11])
            closest_waypoint = int(parts[12])
            track_len = float(parts[13])
            tstamp = Decimal(parts[14])

            iteration = int(episode / episodes_per_iteration) + 1
            df_list.append((iteration, episode, steps, x, y, yaw, steer, throttle,
                            action, reward, done, all_wheels_on_track, progress,
                            closest_waypoint, track_len, tstamp))

        header = ['iteration', 'episode', 'steps', 'x', 'y', 'yaw', 'steer',
                  'throttle', 'action', 'reward', 'done', 'on_track', 'progress',
                  'closest_waypoint', 'track_len', 'timestamp']

        df = pd.DataFrame(df_list, columns=header)
        return df

    @staticmethod
    def load_a_list_of_logs(logs):
        """Loads multiple logs from the list of tuples

        For each file being loaded additional info about the log stream is attached.
        This way one can load multiple simulations for a given period and compare the outcomes.
        This is particularly helpful when comparing multiple evaluations.

        Arguments:
        logs - a list of tuples describing the logs, compatible with the output of
            CloudWatchLogs.download_all_logs

        Returns:
        A pandas dataframe containing all loaded logs data
        """
        full_dataframe = None
        for log in logs:
            eval_data = SimulationLogsIO.load_data(log[0])
            dataframe = SimulationLogsIO.convert_to_pandas(eval_data)
            dataframe['stream'] = log[1]
            if full_dataframe is not None:
                full_dataframe = full_dataframe.append(dataframe)
            else:
                full_dataframe = dataframe

        return full_dataframe.sort_values(
            ['stream', 'episode', 'steps']).reset_index()

    @staticmethod
    def load_pandas(fname, episodes_per_iteration=20):
        """Load from a file directly to pandas dataframe

        Arguments:
        fname - path to the file
        episodes_per_iteration - value of the hyperparameter for a given training

        Returns:
        A pandas dataframe with loaded data
        """
        return SimulationLogsIO.convert_to_pandas(
            SimulationLogsIO.load_data(fname),
            episodes_per_iteration
        )

    @staticmethod
    def normalize_rewards(df):
        """Normalize the rewards to a 0-1 scale

        Arguments:
        df - pandas dataframe with the log data
        """
        from sklearn.preprocessing import MinMaxScaler

        min_max_scaler = MinMaxScaler()
        scaled_vals = min_max_scaler.fit_transform(
            df['reward'].values.reshape(df['reward'].values.shape[0], 1))
        df['reward'] = pd.DataFrame(scaled_vals.squeeze())


class AnalysisUtils:
    """Set of utilities to verify how the training is doing.

    The general purpose is to extract information from the dataframe to visualize it
    in form which allows drawing conclusions from them
    """
    @staticmethod
    def simulation_agg(panda, firstgroup='iteration', add_timestamp=False, is_eval=False):
        """Groups all log data by episodes and other information and returns
        a pandas dataframe with aggregated information

        The aggregated data includes:
        * steps - amount of steps per episode,
        * start_at - starting waypoint
        * progress - how much of the track has been covered
        * throttle - average throttle decision
        * time - how much time elapsed from first to last step
        * reward - how much reward has been aggregated in the iteration
        * time_if_complete - scales time from given progress value to 100% to give
            an idea of what time the car would have if the lap would be completed
        Also if data is for training:
        * new_reward - how much reward would have been aggregated if another
            reward would be used (based on the NewRewardUtils usage)
        * reward_if_complete - scales reward from given progress value to 100% to give
            an idea of what time the car would have if the lap would be completed
        * quintile - which training quintile the episode happened in
            (first 20% of episodes are in 1st, second 20% in 2nd etc.)
        Also if timestamp is requested:
        * timestamp - when given episode ended

        Arguments:
        panda - panda dataframe with simulation data
        firstgroup - first group to group by, by default iteration,
        for multiple log files loaded stream would be preferred
        add_timestamp - whether to add a timestamp, by default False
        is_eval - is data for evaluation (training if False), default: False

        Returns:
        Aggregated dataframe
        """
        grouped = panda.groupby([firstgroup, 'episode'])

        by_steps = grouped['steps'].agg(np.max).reset_index()
        by_start = grouped.first()['closest_waypoint'].reset_index() \
            .rename(index=str, columns={"closest_waypoint": "start_at"})
        by_progress = grouped['progress'].agg(np.max).reset_index()
        by_throttle = grouped['throttle'].agg(np.mean).reset_index()
        by_time = grouped['timestamp'].agg(np.ptp).reset_index() \
            .rename(index=str, columns={"timestamp": "time"})
        by_time['time'] = by_time['time'].astype(float)

        result = by_steps \
            .merge(by_start) \
            .merge(by_progress, on=[firstgroup, 'episode']) \
            .merge(by_time, on=[firstgroup, 'episode'])

        if not is_eval:
            if 'new_reward' not in panda.columns:
                print('new reward not found, using reward as its values')
                panda['new_reward'] = panda['reward']
            by_new_reward = grouped['new_reward'].agg(np.sum).reset_index()
            result = result.merge(by_new_reward, on=[firstgroup, 'episode'])

        result = result.merge(by_throttle, on=[firstgroup, 'episode'])

        if not is_eval:
            by_reward = grouped['reward'].agg(np.sum).reset_index()
            result = result.merge(by_reward, on=[firstgroup, 'episode'])

        result['time_if_complete'] = result['time'] * 100 / result['progress']

        if not is_eval:
            result['reward_if_complete'] = result['reward'] * 100 / result['progress']
            result['quintile'] = pd.cut(result['episode'], 5, labels=[
                                        '1st', '2nd', '3rd', '4th', '5th'])

        if add_timestamp:
            by_timestamp = grouped['timestamp'].agg(np.max).astype(float).reset_index()
            by_timestamp['timestamp'] = pd.to_datetime(by_timestamp['timestamp'], unit='s')
            result = result.merge(by_timestamp, on=[firstgroup, 'episode'])

        return result

    @staticmethod
    def scatter_aggregates(aggregate_df, title=None, is_eval=False):
        """Scatter aggregated data in a set of charts.

        If the data is for evaluation, fewer graphs are shown which makes
        them more readable.
        This set of charts is focused on dependencies other than on iteration/episode.

        Arguments:
        aggregate_df - aggregated data
        title - what title to give to the charts (None by default)
        is_eval - is it evaluation data (training if False), by default False
        """
        fig, axes = plt.subplots(nrows=2 if is_eval else 3,
                                 ncols=2 if is_eval else 3, figsize=[15, 11])
        if title:
            fig.suptitle(title)
        if not is_eval:
            aggregate_df.plot.scatter('time', 'reward', ax=axes[0, 2])
            aggregate_df.plot.scatter('time', 'new_reward', ax=axes[1, 2])
            aggregate_df.plot.scatter('start_at', 'reward', ax=axes[2, 2])
            aggregate_df.plot.scatter('start_at', 'progress', ax=axes[2, 0])
            aggregate_df.plot.scatter('start_at', 'time_if_complete', ax=axes[2, 1])
        aggregate_df.plot.scatter('time', 'progress', ax=axes[0, 0])
        aggregate_df.hist(column=['time'], bins=20, ax=axes[1, 0])
        aggregate_df.plot.scatter('time', 'steps', ax=axes[0, 1])
        aggregate_df.hist(column=['progress'], bins=20, ax=axes[1, 1])

        plt.show()
        plt.clf()

    @staticmethod
    def scatter_by_groups(aggregate_df, group_category='quintile', title=None):
        """Visualise aggregated training data grouping them by a given category.

        Takes the aggregated dataframe and groups it by category.
        By default quintile is being used which means all episodes are divided into
        five buckets by time. This lets you observe how the training progressed
        with time

        Arguments:
        aggregate_df - aggregated dataframe
        group_category - what to group the data by, default: quintile
        title - what title to put over the charts, default: None
        """
        grouped = aggregate_df.groupby(group_category)
        groupcount = len(grouped.groups.keys())

        fig, axes = plt.subplots(nrows=groupcount, ncols=4, figsize=[15, 15])

        if title:
            fig.suptitle(title)

        row = 0
        for _, group in grouped:
            group.plot.scatter('time', 'reward', ax=axes[row, 0])
            group.plot.scatter('time', 'new_reward', ax=axes[row, 1])
            group.hist(column=['time'], bins=20, ax=axes[row, 2])
            axes[row, 3].set(xlim=(0, 100))
            group.hist(column=['progress'], bins=20, ax=axes[row, 3])
            row += 1

        plt.show()
        plt.clf()

    @staticmethod
    def analyze_training_progress(aggregates, title=None):
        """Analyze training progress based on iterations

        Most of the charts have iteration as the x axis which shows how rewards,
        times, progress and others have changed in time iteration by iteration.
        The graphs present:
        * mean reward with standard deviation
        * total reward
        * mean time with standard deviation
        * mean time for completed laps
        * mean progress with standard deviation
        * completion rate (ratio of complete laps to all episodes in iteration, 0-1)

        Arguments:
        aggregates - aggregated dataframe to analyze
        title - what title to put over the charts, default: None
        """
        aggregates['complete'] = np.where(aggregates['progress'] == 100, 1, 0)

        grouped = aggregates.groupby('iteration')

        reward_per_iteration = grouped['reward'].agg([np.mean, np.std]).reset_index()
        time_per_iteration = grouped['time'].agg([np.mean, np.std]).reset_index()
        progress_per_iteration = grouped['progress'].agg([np.mean, np.std]).reset_index()

        complete_laps = aggregates[aggregates['progress'] == 100.0]
        complete_grouped = complete_laps.groupby('iteration')

        complete_times = complete_grouped['time'].agg([np.mean, np.min, np.max]).reset_index()

        total_completion_rate = complete_laps.shape[0] / aggregates.shape[0]

        complete_per_iteration = grouped['complete'].agg([np.mean]).reset_index()

        print('Number of episodes = ', np.max(aggregates['episode']))
        print('Number of iterations = ', np.max(aggregates['iteration']))

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=[15, 15])

        if title:
            fig.suptitle(title)

        AnalysisUtils.plot(axes[0, 0], reward_per_iteration, 'iteration', 'Iteration',
                           'mean', 'Mean reward', 'Rewards per Iteration')
        AnalysisUtils.plot(axes[1, 0], reward_per_iteration, 'iteration',
                           'Iteration', 'std', 'Std dev of reward', 'Dev of reward')
        AnalysisUtils.plot(axes[2, 0], aggregates, 'episode', 'Episode', 'reward', 'Total reward')

        AnalysisUtils.plot(axes[0, 1], time_per_iteration, 'iteration',
                           'Iteration', 'mean', 'Mean time', 'Times per Iteration')
        AnalysisUtils.plot(axes[1, 1], time_per_iteration, 'iteration',
                           'Iteration', 'std', 'Std dev of time', 'Dev of time')
        if complete_times.shape[0] > 0:
            AnalysisUtils.plot(axes[2, 1], complete_times, 'iteration', 'Iteration',
                               'mean', 'Time', 'Mean completed laps time')

        AnalysisUtils.plot(axes[0, 2], progress_per_iteration, 'iteration', 'Iteration', 'mean',
                           'Mean progress', 'Progress per Iteration')
        AnalysisUtils.plot(axes[1, 2], progress_per_iteration, 'iteration',
                           'Iteration', 'std', 'Std dev of progress', 'Dev of progress')
        AnalysisUtils.plot(axes[2, 2], complete_per_iteration, 'iteration', 'Iteration', 'mean',
                           'Completion rate', 'Completion rate (avg: %s)' % total_completion_rate)

        plt.show()
        plt.clf()

    @staticmethod
    def plot(ax, df, xval, xlabel, yval, ylabel, title=None):
        """plot the data and put in the right place on charts image

        Arguments:
        ax - plot axes
        df - dataframe to extract data from
        xval - which data to extract for x axis
        xlabel - what label to give the x axis
        yval - which data to extract for y axis
        ylabel - what label to give the y axis
        title - what title to put over the chart, default: None
        """
        df.plot.scatter(xval, yval, ax=ax, s=5, alpha=0.7)
        if title:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        plt.grid(True)


class PlottingUtils:
    """Utilities to visualise track and the episodes
    """
    @staticmethod
    def print_border(ax, track: Track, color='lightgrey'):
        """Print track borders on the chart

        Arguments:
        ax - axes to plot borders on
        track - the track info to plot
        color - what color to plot the border in, default: lightgrey
        """
        line = LineString(track.center_line)
        PlottingUtils._plot_coords(ax, line)
        PlottingUtils._plot_line(ax, line, color)

        line = LineString(track.inner_border)
        PlottingUtils._plot_coords(ax, line)
        PlottingUtils._plot_line(ax, line, color)

        line = LineString(track.outer_border)
        PlottingUtils._plot_coords(ax, line)
        PlottingUtils._plot_line(ax, line, color)

    @staticmethod
    def plot_selected_laps(sorted_idx, df, track: Track, section_to_plot="episode"):
        """Plot n laps in the training, referenced by episode ids

        Arguments:
        sorted_idx - a datagram with ids to be plotted or a list of ids
        df - a datagram with all data
        track - track info for plotting
        secton_to_plot - what section of data to plot - episode/iteration
        """

        ids = sorted_idx

        if type(sorted_idx) is not list:
            ids = sorted_idx[section_to_plot].unique().tolist()

        n_laps = len(ids)

        fig = plt.figure(n_laps, figsize=(12, n_laps * 10))
        for i in range(n_laps):
            idx = ids[i]

            data_to_plot = df[df[section_to_plot] == idx]

            ax = fig.add_subplot(n_laps, 1, i + 1)

            ax.axis('equal')

            PlottingUtils.print_border(ax, track, color='cyan')

            data_to_plot.plot.scatter('x', 'y', ax=ax, s=10, c='blue')

        plt.show()
        plt.clf()

        # return fig

    @staticmethod
    def plot_evaluations(evaluations, track: Track, graphed_value='throttle'):
        """Plot graphs for evaluations
        """
        from math import ceil

        streams = evaluations.sort_values(
            'timestamp', ascending=False).groupby('stream', sort=False)

        for _, stream in streams:
            episodes = stream.groupby('episode')
            ep_count = len(episodes)

            rows = ceil(ep_count / 3)
            columns = min(ep_count, 3)

            fig, axes = plt.subplots(rows, columns, figsize=(7*columns, 5*rows))
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=7.0)

            for id, episode in episodes:
                if rows == 1:
                    ax = axes[id % 3]
                elif columns == 1:
                    ax = axes[int(id/3)]
                else:
                    ax = axes[int(id / 3), id % 3]

                PlottingUtils.plot_grid_world(
                    episode, track, graphed_value, ax=ax)

            plt.show()
            plt.clf()

    @staticmethod
    def plot_grid_world(
        episode_df,
        track: Track,
        graphed_value='throttle',
        min_progress=None,
        ax=None
    ):
        """Plot a scaled version of lap, along with throttle taken a each position
        """

        episode_df.loc[:, 'distance_diff'] = ((episode_df['x'].shift(1) - episode_df['x']) ** 2 + (
            episode_df['y'].shift(1) - episode_df['y']) ** 2) ** 0.5

        distance = np.nansum(episode_df['distance_diff'])
        lap_time = np.ptp(episode_df['timestamp'].astype(float))
        velocity = distance / lap_time
        average_throttle = np.nanmean(episode_df['throttle'])
        progress = np.nanmax(episode_df['progress'])

        if not min_progress or progress > min_progress:

            distance_lap_time = 'Distance, progress, lap time = %.2f m, %.2f %%, %.2f s' % (
                distance, progress, lap_time
            )
            throttle_velocity = 'Average throttle, velocity = %.2f (Gazebo), %.2f m/s' % (
                average_throttle, velocity
            )

            fig = None
            if ax is None:
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(1, 1, 1)

            ax.set_facecolor('midnightblue')

            line = LineString(track.inner_border)
            PlottingUtils._plot_coords(ax, line)
            PlottingUtils._plot_line(ax, line)

            line = LineString(track.outer_border)
            PlottingUtils._plot_coords(ax, line)
            PlottingUtils._plot_line(ax, line)

            episode_df.plot.scatter('x', 'y', ax=ax, s=3, c=graphed_value,
                                    cmap=plt.get_cmap('plasma'))

            subtitle = '%s%s\n%s\n%s' % (
                ('Stream: %s, ' % episode_df['stream'].iloc[0]
                 ) if 'stream' in episode_df.columns else '',
                datetime.fromtimestamp(episode_df['timestamp'].iloc[0]),
                distance_lap_time,
                throttle_velocity)
            ax.set_title(subtitle)

            if fig:
                plt.show()
                plt.clf()

    @staticmethod
    def plot_track(df, track: Track, value_field="reward", margin=1, cmap="hot"):
        """Plot track with dots presenting the rewards for steps
        """
        track_size = (np.asarray(track.size()) + 2*margin).astype(int) * 100
        track_img = np.zeros(track_size).transpose()

        x_coord = 0
        y_coord = 1

        # compensation moves car's coordinates in logs to start at 0 in each dimention
        x_compensation = df['x'].min()
        y_compensation = df['y'].min()

        for _, row in df.iterrows():
            x = int((row["x"] - x_compensation + margin) * 100)
            y = int((row["y"] - y_compensation + margin) * 100)

            # clip values that are off track
            if y >= track_size[y_coord]:
                y = track_size[y_coord] - 1

            if x >= track_size[x_coord]:
                x = track_size[x_coord] - 1

            track_img[y, x] = row[value_field]

        fig = plt.figure(1, figsize=(12, 16))
        ax = fig.add_subplot(111)

        shifted_track = Track("shifted_track", (track.waypoints -
                                                [x_compensation, y_compensation]*3 + margin) * 100)

        PlottingUtils.print_border(ax, shifted_track)

        plt.title("Reward distribution for all actions ")
        plt.imshow(track_img, cmap=cmap, interpolation='bilinear', origin="lower")

        plt.show()
        plt.clf()

    @staticmethod
    def plot_trackpoints(track: Track, annotate_every_nth=1):
        _, ax = plt.subplots(figsize=(20, 10))
        PlottingUtils.plot_points(ax, track.center_line, annotate_every_nth)
        PlottingUtils.plot_points(ax, track.inner_border, annotate_every_nth)
        PlottingUtils.plot_points(ax, track.outer_border, annotate_every_nth)
        ax.axis('equal')

        return ax

    @staticmethod
    def plot_points(ax, points, annotate_every_nth=1):
        ax.scatter(points[:-1, 0], points[:-1, 1], s=1)
        for i, p in enumerate(points):
            if i % annotate_every_nth == 0:
                ax.annotate(i, (p[0], p[1]))

    @staticmethod
    def _plot_coords(ax, ob):
        x, y = ob.xy
        ax.plot(x, y, '.', color='#999999', zorder=1)

    @staticmethod
    def _plot_bounds(ax, ob):
        x, y = zip(*list((p.x, p.y) for p in ob.boundary))
        ax.plot(x, y, '.', color='#000000', zorder=1)

    @staticmethod
    def _plot_line(ax, ob, color='cyan'):
        x, y = ob.xy
        ax.plot(x, y, color=color, alpha=0.7, linewidth=3, solid_capstyle='round',
                zorder=2)


class EvaluationUtils:
    @staticmethod
    def analyse_single_evaluation(eval_df, track: Track,
                                  min_progress=None):
        """Plot all episodes of a single evaluation
        """
        episodes = eval_df.groupby('episode').groups
        for e in episodes:
            PlottingUtils.plot_grid_world(
                eval_df[eval_df['episode'] == e], track, min_progress=min_progress)

    @staticmethod
    def analyse_multiple_race_evaluations(logs, track: Track, min_progress=None):
        for log in logs:
            EvaluationUtils.analyse_single_evaluation(
                SimulationLogsIO.load_pandas(log[0]), track, min_progress=min_progress)

    @staticmethod
    def download_and_analyse_multiple_race_evaluations(
        log_folder,
        track: Track,
        not_older_than=None,
        older_than=None,
        log_group='/aws/deepracer/leaderboard/SimulationJobs',
        min_progress=None
    ):
        """Download and analyse multiple race evaluations

        Arguments:
        log_folder - where should the logs be stored
        track - track to plot
        not_older_than - the oldest date at which a stream has to have
            at least one message to be downloaded; unlimited if not
            provided; ISO-8601 compliant date string. Example:
            "2020-02-20 02:02 UTC"
        older_than - the most recent date at which a stream has to have
            at least one message to be downloaded; unlimited if not
            provided; ISO-8601 compliant date string. Example:
            "2020-02-20 02:02 UTC"
        log_group - which group to look in for the streams; default:
            '/aws/deepracer/leaderboard/SimulationJobs'
        min_progress - minimum progres for an episode to be plotted, default: None

        """
        logs = cw.download_all_logs("%s/deepracer-eval-" % log_folder,
                                    log_group, not_older_than, older_than)

        EvaluationUtils.analyse_multiple_race_evaluations(
            logs, track, min_progress=min_progress)


class NewRewardUtils:
    """New reward testing utility

    This will not give you a newly trained model, but based on the logs you will
    be able to check how your training would be marked with an alternative reward
    function.

    The reward needs to be wrapped in an object, so the absolute minimum would be:

    class Reward:
        def __init__(self, verbose=False):
            self.verbose = verbose

        def reward_function(self, params):
            #reward calculation happens here
            return float(reward)

    You are allowed to use fields to hold state. If your reward function file also
    contains:

    reward = Reward()

    def reward_function(params):
        return reward.reward_function(params)

    you can use it as is in the AWS DeepRacer Console.
    """
    @staticmethod
    def df_to_params(df_row, waypoints):
        """Convert log data to parameters to be passed to be passed to the reward function

        Arguments:
        df_row - single row to be converted to parameters
        waypoints - waypoints to put into the parameters

        Returns:
        A dictionary of parameters
        """
        from ..tracks.track_utils import GeometryUtils as gu
        waypoint = df_row['closest_waypoint']
        before = waypoint - 1
        if waypoints[waypoint].tolist() == waypoints[before].tolist():
            before -= 1
        after = (waypoint + 1) % len(waypoints)

        if waypoints[waypoint].tolist() == waypoints[after].tolist():
            after = (after + 1) % len(waypoints)

        current_location = np.array([df_row['x'], df_row['y']])

        closest_point = gu.get_a_point_on_a_line_closest_to_point(
            waypoints[before],
            waypoints[waypoint],
            [df_row['x'], df_row['y']]
        )

        if gu.is_point_roughly_on_the_line(
            waypoints[before],
            waypoints[waypoint],
            closest_point[0], closest_point[1]
        ):
            closest_waypoints = [before, waypoint]
        else:
            closest_waypoints = [waypoint, after]

            closest_point = gu.get_a_point_on_a_line_closest_to_point(
                waypoints[waypoint],
                waypoints[after],
                [df_row['x'], df_row['y']]
            )

        params = {
            'x': df_row['x'],
            'y': df_row['y'],
            'speed': df_row['throttle'],
            'steps': df_row['steps'],
            'progress': df_row['progress'],
            'heading': df_row['yaw'] * 180 / 3.14,
            'closest_waypoints': closest_waypoints,
            'steering_angle': df_row['steer'] * 180 / 3.14,
            'waypoints': waypoints,
            'distance_from_center':
                gu.get_vector_length(
                    (
                        closest_point -
                        current_location
                    )),
            'timestamp': df_row['timestamp'],
            # TODO I didn't need them yet. DOIT
            'track_width': 0.60,
            'is_left_of_center': None,
            'all_wheels_on_track': True,
            'is_reversed': False,
        }

        return params

    @staticmethod
    def new_reward(panda, center_line, reward_module, verbose=False):
        """Calculate new reward for each step and add to the dataframe

        Arguments:
        panda - the dataframe to replay
        center_line - waypoints along the track's center_line
        reward_module - which module to load for calculation
        verbose - should the reward function print extra info (if there is such option)
        """
        import importlib
        importlib.invalidate_caches()
        rf = importlib.import_module(reward_module)
        importlib.reload(rf)

        reward = rf.Reward(verbose=verbose)

        new_rewards = []
        for _, row in panda.iterrows():
            new_rewards.append(
                reward.reward_function(NewRewardUtils.df_to_params(row, center_line)))

        panda['new_reward'] = new_rewards


class ActionBreakdownUtils:
    """Utilities to perform action breakdown analysis
    """
    @staticmethod
    def determine_action_names(df):
        keys = sorted(df.groupby(["action", "steer", "throttle"]).groups.keys(), key=lambda x: x[0])

        return ["A:%d S:%s, T:%s%s" % (
            key[0],
            key[2],
            abs(key[1]),
            " LEFT" if key[1] > 0 else " RIGHT" if key[1] < 0 else ""
        ) for key in filter(lambda key: key[2] > 0, keys)]

    @staticmethod
    def _make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                          edgecolor='r', alpha=0.3):
        # Create list for all the error patches
        errorboxes = []

        # Loop over data points; create box from errors at each point
        for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
            rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
            errorboxes.append(rect)

        # Create patch collection with specified colour/alpha
        pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                             edgecolor=edgecolor)

        # Add collection to axes
        ax.add_collection(pc)

        return 0

    @staticmethod
    def action_breakdown(
        df,
        track: Track,
        iteration_ids=None,
        episode_ids=None,
        track_breakdown=None,
        action_names=None,
        min_reward=0.0
    ):
        """Visualise action breakdown for the simulation data

        Arguments:
        df - dataframe to visualise
        track - track info to plot
        iteration_ids - which episodes to visualise
        track_breakdown - interesting sections of the track to show,
            default: None (no sections highlighted)
        action_names - how to call the actions; default: None (names will be generated)
        """
        if not action_names:
            action_names = ActionBreakdownUtils.determine_action_names(df)

        fig = plt.figure(figsize=(16, len(action_names)*6))

        if iteration_ids is not None and type(iteration_ids) is not list:
            iteration_ids = [iteration_ids]

        if episode_ids is not None and type(episode_ids) is not list:
            episode_ids = [episode_ids]

        wpts_array = track.center_line

        # Slice the data frame to get all episodes in selected iterations
        df_iter = df[df['iteration'].isin(iteration_ids)] if iteration_ids is not None else df

        # Slice the data frame to get all episodes in list
        df_iter = df[df['episode'].isin(episode_ids)] if episode_ids is not None else df

        for idx in range(len(action_names)):
            ax = fig.add_subplot(len(action_names), 2, 2 * idx + 1)
            PlottingUtils.print_border(
                ax,
                track
            )

            df_slice = df_iter[df_iter['action'] == idx]
            df_slice = df_slice[df_slice['reward'] >= min_reward]

            ax.plot(df_slice['x'], df_slice['y'], 'b.')

            if track_breakdown:
                for idWp in track_breakdown.vert_lines:
                    ax.text(wpts_array[idWp][0],
                            wpts_array[idWp][1] + 0.2,
                            str(idWp),
                            bbox=dict(facecolor='red', alpha=0.5))

            # ax.set_title(str(log_name_id) + '-' + str(iter_num) + ' w rew >= '+str(th))
            ax.set_ylabel(action_names[idx])

            # calculate action way point distribution
            action_waypoint_distribution = list()
            for idWp in range(len(wpts_array)):
                action_waypoint_distribution.append(
                    len(df_slice[df_slice['closest_waypoint'] == idWp]))

            ax = fig.add_subplot(len(action_names), 2, 2 * idx + 2)

            if track_breakdown:
                # Call function to create error boxes
                _ = ActionBreakdownUtils._make_error_boxes(
                    ax,
                    track_breakdown.segment_x,
                    track_breakdown.segment_y,
                    track_breakdown.segment_xerr,
                    track_breakdown.segment_yerr
                )

                for tt in range(len(track_breakdown.track_segments)):
                    ax.text(track_breakdown.track_segments[tt][0],
                            track_breakdown.track_segments[tt][1],
                            track_breakdown.track_segments[tt][2])

            ax.bar(np.arange(len(wpts_array)), action_waypoint_distribution)
            ax.set_xlabel('waypoint')
            ax.set_ylabel('# of actions')
            ax.legend([action_names[idx]])
            ax.set_ylim((0, 150))

        plt.show()
        plt.clf()
