from matplotlib import animation
import matplotlib.pyplot as plt
import argparse
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import tensorflow as tf
import os
import gym 
import agent
import ppo

# this code is modified from https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


parser = argparse.ArgumentParser(description="visualize ppo policy")
parser.add_argument('load_model', type=str, help="path to model to load")
args = parser.parse_args()
savepath = args.load_model


with tf.device('/device:CPU:0'):
  actor = tf.saved_model.load(savepath)

  #Make gym env
  env = ppo.makeEnv()
  
  #Run the env
  state = env.reset()
  frames = []
  for t in range(3000):
      print(t)
      #Render to frames buffer
      state = np.expand_dims(state._force(), 0)
      frames.append(env.render(mode="rgb_array"))
      policy = actor.policy(state)
      action, _ = actor.act(policy)
      state, _, done, _ = env.step(action)
      if done:
          print("env ended!")
          break
  env.close()
  save_frames_as_gif(frames)
