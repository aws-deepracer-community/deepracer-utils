# Examples

In the lastest version of Deepracer Utils the foundation class to use is `DeepRacerLog`. This class will wrap around a model folder, either stored locally or in an S3 bucket. The class supports console logs, as well as DRfC logs for single and multiple workers.


## Training

### Local

The basic example covers a model director downloaded from the console ~~or stored in a minio folder~~, and where the training simtrace files, the `model_metadata.json` and `hyperparameters.json` files are available.

```
from deepracer.logs import (AnalysisUtils, DeepRacerLog)

drl = DeepRacerLog(model_folder='./deepracer/logs/sample-console-logs')
drl.load_training_trace()
df = drl.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df)
```

You will end up with two objects:
* `df` being a pandas dataframe which stores each and every step. This is the data to use if you want to plot the trajectory of the car, to see the distribution of actions and rewards.
* `simulation_agg` is a dataframe which is aggregated to episode level, providing information such as episode duration, overall episode progress, distance travelled etc.

### S3 Bucket

If you have stored your models in an S3 bucket, then you need to configure a specific `S3FileHandler` to be used rather than the default `FSFileHandler`.

```
from deepracer.logs import (AnalysisUtils, DeepRacerLog, S3FileHandler)

fh = S3FileHandler(bucket="<my_bucket>",
                    prefix="<my_prefix>")
drl = DeepRacerLog(filehandler=fh)
drl.load_training_trace()
df = drl.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df)
```

Outcomes are the same as for the local model.

### Local S3 / Minio

If you use a local minio installation you need to add the `profile` and `s3_endpoint_url` parameters to be able to log into minio. Typical values are `minio` and `http://localhost:9000`.

```
from deepracer.logs import (AnalysisUtils, DeepRacerLog, S3FileHandler)

fh = S3FileHandler(bucket="<my_bucket>", prefix="<my_prefix>",
                   profile="<awscli_profile>", s3_endpoint_url="<minio_url>")
drl = DeepRacerLog(filehandler=fh)
drl.load_training_trace()
df = drl.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df)
```

### Robomaker Logs

Instead of loading in the trace logs one can also load in the Robomaker log. This should not have an impact on the output.

```
from deepracer.logs import (AnalysisUtils, DeepRacerLog, S3FileHandler)

fh = S3FileHandler(bucket="<my_bucket>",
                    prefix="<my_prefix>")
drl = DeepRacerLog(filehandler=fh)
drl.load_robomaker_logs()
df = drl.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df)
```

## Evaluation

It is also possible to read in evaluation logs. Where training only contains one set of logs, evaluations can contain multiple logs, either as trace files or as Robomaker logs. The class will load in all the files that it finds, and separate them with a time-stamp in the `stream` column.

```
from deepracer.logs import (AnalysisUtils, DeepRacerLog, S3FileHandler)

fh = S3FileHandler(bucket="<my_bucket>",
                    prefix="<my_prefix>")
drl = DeepRacerLog(filehandler=fh)
drl.load_evaluation_trace()
df = drl.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
```

Note the additional parameters to enable proper aggregation of the data.

```
from deepracer.logs import (AnalysisUtils, DeepRacerLog, S3FileHandler, LogType)

fh = S3FileHandler(bucket="<my_bucket>",
                    prefix="<my_prefix>")
drl = DeepRacerLog(filehandler=fh)
drl.load_robomaker_logs(type=LogType.EVALUATION)
df = drl.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
```

## Leaderboard Submissions

Similarily to the Evaluation logs it is possible to load in leaderboard submissions through exporting the logs of a model to S3 in the console. In the case of such export all evaluations and leaderboard submissions are exported.

Leaderboard submissions are only available as Robomaker logs.

```
from deepracer.logs import (AnalysisUtils, DeepRacerLog, S3FileHandler, LogType)

fh = S3FileHandler(bucket="<my_bucket>",
                    prefix="<my_prefix>")
drl = DeepRacerLog(filehandler=fh)
drl.load_robomaker_logs(type=LogType.LEADERBOARD)
df = drl.dataframe()

simulation_agg = AnalysisUtils.simulation_agg(df, firstgroup='stream', is_eval=True)
```
