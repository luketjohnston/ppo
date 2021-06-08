import matplotlib
import argparse
from time import sleep
matplotlib.use('tkagg')

import tensorflow as tf
import os
import gym

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import agent
import ppo


parser = argparse.ArgumentParser(description="visualize ppo policy")
parser.add_argument('load_model', type=str, help="path to model to load")
args = parser.parse_args()
savepath = args.load_model

with tf.device('/device:CPU:0'):
  actor = tf.saved_model.load(savepath)
  
  # make environment
  env = ppo.makeEnv()
  state = env.reset()
  
  
  while True:
  
   
    state = np.expand_dims(state._force(), 0)
    policy =  actor.policy(state)
    action, _ = actor.act(policy)
  
    sleep(0.0416)
    state, _, done, _ = env.step(action)
  
    if done: env.reset()
  
    env.render()
  
  
  env.close()
  
  
  
