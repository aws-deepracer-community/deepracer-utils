import os
import warnings
import pytest
import json
import glob

import cv2
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.io.gfile import GFile

from deepracer.model import load_session, visualize_gradcam_discrete_ppo, rgb2gray 

class TestTrainingLogs:

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self, tmpdir):
        yield

    def test_load_model(self):
        model_path='./deepracer/model/sample-model/model'
        iterations = [15, 30, 48]
        img_selection = './deepracer/model/sample-model/pictures/*.png'

        with open("{}/model_metadata.json".format(model_path),"r") as jsonin:
            model_metadata=json.load(jsonin)
        my_sensor = [sensor for sensor in model_metadata['sensor'] if sensor != "LIDAR"][0]

        assert "FRONT_FACING_CAMERA" == my_sensor

        action_names = []
        degree_sign= u'\N{DEGREE SIGN}'
        for action in model_metadata['action_space']:
            action_names.append(str(action['steering_angle'])+ degree_sign + " "+"%.1f"%action["speed"])
        assert 12 == len(action_names)

        picture_files = sorted(glob.glob(img_selection))
        assert 3 == len(picture_files)

        model_inference = []
        models_file_path = []

        for n in iterations:
            models_file_path.append("{}/model_{}.pb".format(model_path,n))        

        for model_file in models_file_path:
            model, obs, model_out = load_session(model_file, my_sensor, False)
            arr = []
            for f in picture_files[:]:
                img = cv2.imread(f)
                img = cv2.resize(img, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
                img_arr = np.array(img)
                img_arr = rgb2gray(img_arr)
                img_arr = np.expand_dims(img_arr, axis=2)
                current_state = {"observation": img_arr} #(1, 120, 160, 1)
                y_output = model.run(model_out, feed_dict={obs:[img_arr]})[0]
                arr.append (y_output)
                
            model_inference.append(arr)
            model.close()
            tf.reset_default_graph()

        heatmaps = []
        view_models = models_file_path[1:3]

        for model_file in view_models:
            model, obs, model_out = load_session(model_file, my_sensor, False)
            arr = []
            for f in picture_files:
                img = cv2.imread(f)
                img = cv2.resize(img, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
                heatmap = visualize_gradcam_discrete_ppo(model, img, category_index=0, num_of_actions=len(action_names))
                heatmaps.append(heatmap)    

            tf.reset_default_graph()