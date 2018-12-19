import tensorflow as tf
import json
import os


config_path = os.path.dirname(os.path.realpath(__file__))
params_path = os.path.join(config_path, 'params.json')
print(params_path)

with open(params_path) as param:
    params_dict = json.load(param)
Config = tf.contrib.training.HParams(**params_dict)
print(Config)