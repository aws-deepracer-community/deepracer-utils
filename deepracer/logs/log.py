import glob
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

class DeepRacerConsoleLog(DeepRacerLog):
    def __init__(self, model_folder):
        super().__init__()

        self.model_folder = model_folder
        
        self.iter_count = {}

        self.df = None

    def load_model_iteration(self, model_iteration):
        dfs = []

        dfs = [self.load_worker(model_iteration, worker_id) for worker_id in worker_ids]
        if len(dfs) == 0:

            return None

        df = pd.concat(dfs, ignore_index=True)
        df["model_iteration"] = "rl-deepracer-{}".format(model_iteration)

        return df


    def load(self):
        # Load all available iteration files.
        model_iterations = glob.glob(os.path.join(self.model_folder, "**", "**", "training-simtrace", "*-iteration.csv"))
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
            missing = set(self.hyperparam_keys)
            robomaker_log = glob.glob(os.path.join(self.model_folder, "**", "training", "*-robomaker.log"))[0]
            hyperparameters = {}
        except:
            raise Exception("Could not find robomaker log!")
            
        with open(robomaker_log, 'r') as f:
            for line in f.read().splitlines():
                for key in missing:
                    # TODO: Fig regex
#                     match = re.search(r'"{}": (.*)'.format(key), line)

                    if match:
                        value = match.group(1).replace(",", "")

                        if value.isnumeric():
                            value = float(value)

                        hyperparameters[key] = value
                        missing.remove(key)
                        break

                if len(missing) == 0:
                    break

            return hyperparameters


