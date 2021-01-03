import glob
import json
import os
import pandas as pd
import re
from joblib import Parallel, delayed


class DeepRacerLog:
    def __init__(self):
        # Column names we support in the CSV file.
        self.col_names = [
            "episode",
            "steps",
            "x",
            "y",
            "heading",
            "steering_angle",
            "speed",
            "action",
            "reward",
            "done",
            "all_wheels_on_track",
            "progress",
            "closest_waypoint",
            "track_len",
            "tstamp",
            "episode_status",
        ]
        self.hyperparam_keys = [
            "batch_size",
            "beta_entropy",
            "e_greedy_value",
            "epsilon_steps",
            "exploration_type",
            "loss_type",
            "lr",
            "num_episodes_between_training",
            "num_epochs",
            "stack_size",
            "term_cond_avg_score",
            "term_cond_max_episodes"
        ]

    def load(self):
        """Method that loads a DeepRacer log into a dataframe.

        Raises:
            NotImplementedError This class should not be used.
        """
        raise NotImplementedError("Implement this in a subclass!")

    def dataframe(self):
        """Method that provides the dataframe for analysis of this log.

        Raises:
            NotImplementedError: This class should not be used.
        """
        raise NotImplementedError("Implement this in a subclass!")

    def hyperparameters(self):
        """Method that provides the hyperparameters for this log.

        Raises:
            NotImplementedError: This class should not be used.
        """
        raise NotImplementedError("Implement this in a subclass!")

    def action_space(self):
        """Method that provides the action space for this log.

        Raises:
            NotImplementedError: This class should not be used.
        """
        raise NotImplementedError("Implement this in a subclass!")

    def agent_and_network(self):
        """Method that provides the agent and network information for this log.
        Resulting dictionary includes the name of environment used,
        list of sensors and type of network.

        Raises:
            NotImplementedError: This class should not be used.
        """
        raise NotImplementedError("Implement this in a subclass!")


class DeepRacerConsoleLog(DeepRacerLog):
    def __init__(self, model_folder):
        super().__init__()

        self.model_folder = model_folder

        self.iter_count = {}

        self.df = None

    def load(self):
        # Load all available iteration files.
        model_iterations = glob.glob(os.path.join(
            self.model_folder, "**", "**", "training-simtrace", "*-iteration.csv"))
        print(self.model_folder)

        def read_csv(path, iteration):
            df = pd.read_csv(path, names=self.col_names, header=0)

            df["iteration"] = iteration

            return df

        dfs = Parallel(n_jobs=-1, prefer="threads")(
            delayed(read_csv)(path, i) for i, path in enumerate(model_iterations)
        )

        if len(dfs) == 0:
            return

        # Merge into single large DataFrame
        df = pd.concat(dfs, ignore_index=True)

        self.df = df

    def dataframe(self):
        if self.df is None:
            raise Exception("Model not loaded, call load() before requesting a dataframe.")

        return self.df

    def hyperparameters(self):
        try:
            robomaker_log = glob.glob(os.path.join(
                self.model_folder, "**", "training", "*-robomaker.log"))[0]
        except Exception as e:
            raise Exception("Could not find robomaker log!") from e

        outside_hyperparams = True
        hyperparameters_string = ""
        with open(robomaker_log, 'r') as f:
            for line in f:
                if outside_hyperparams:
                    if "Using the following hyper-parameters" in line:
                        outside_hyperparams = False
                else:
                    hyperparameters_string += line
                    if "}" in line:
                        break

        return json.loads(hyperparameters_string)

    def action_space(self):
        try:
            robomaker_log = glob.glob(os.path.join(
                self.model_folder, "**", "training", "*-robomaker.log"))[0]
        except Exception as e:
            raise Exception("Could not find robomaker log!") from e

        with open(robomaker_log, 'r') as f:
            for line in f:
                if "Action space from file: " in line:
                    return json.loads(line[24:].replace("'", '"'))

    def agent_and_network(self):
        try:
            robomaker_log = glob.glob(os.path.join(
                self.model_folder, "**", "training", "*-robomaker.log"))[0]
        except Exception as e:
            raise Exception("Could not find robomaker log!") from e

        with open(robomaker_log, 'r') as f:
            result = {}
            for line in f:
                if " * /WORLD_NAME: " in line:
                    result["world"] = line[:-1].split(" ")[-1]
                elif "Sensor list ['" in line:
                    data = line[:-1].split(", ")
                    result["sensor_list"] = json.loads(data[0].split(" ")[-1].replace("'", '"'))
                    for info in data[1:]:
                        data_bits = info.split(" ")
                        result[data_bits[0]] = data_bits[1]
                    return result
