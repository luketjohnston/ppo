import tensorflow as tf
import argparse
import multiprocessing as mp
from datetime import datetime
import matplotlib
import timeit
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import random
import pickle 
import os
from skimage.transform import resize
from tensorflow.keras import Model
import gym
from contextlib import ExitStack


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import agent
from vecenv import VecEnv # not parallelized yet

from atari_wrappers import WarpFrame, NoopResetEnv, EpisodicLifeEnv, ClipRewardEnv, FrameStack, ScaledFloatFrame, MaxAndSkipEnv, FireResetEnv

def makeEnv():
  env = gym.make(agent.ENVIRONMENT)
  if not agent.ENVIRONMENT == 'CartPole-v1' and not agent.ENVIRONMENT == 'Acrobot-v1':
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)

    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
      env = FireResetEnv(env)
    env = WarpFrame(env, 84, 84)
    env = ScaledFloatFrame(env) # negates optimization of LazyFrames in FrameStack
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
  return env


LOAD_SAVE = True
PROFILE = False

GAN_REWARD_WEIGHT = 0.01
SAVE_CYCLES = 20
ENVS = agent.ENVS
ROLLOUT_LEN = agent.ROLLOUT_LEN
MINIBATCH_SIZE = agent.MINIBATCH_SIZE
MINIBATCHES = ENVS * ROLLOUT_LEN // MINIBATCH_SIZE 
FRAMES_PER_CYCLE = ENVS * ROLLOUT_LEN
EPOCHS = agent.EPOCHS
WGAN_TRAINING_CYCLES = 1
CRITIC_BATCHES = 20
GEN_BATCHES = 1

PARAM_UPDATES_PER_CYCLE = MINIBATCHES * EPOCHS


''' 
given vecenv and actor, 
computes the generalized advantage estimate for a rollout of n steps.

returns:
[states, rewards, actions, dones, act_probs], episode_rewards
NOTE: states has length n+1 because it includes the final state (after the
rollout is finished). This is used to compute the one-step TD-error for
the last rollout step

episode_rewards is a list of total reward for any episodes that terminated during
the rollout.
'''
def getRollout(vecenv, n, actor):

  # + 1 to hold extra state after last transition, to use to get final one-step td estimate
  states = np.zeros([n + 1, vecenv.nenvs] + agent.INPUT_SHAPE, np.float32)
  rewards = np.zeros([n, vecenv.nenvs])
  actions = np.zeros([n, vecenv.nenvs], dtype=np.int)
  dones = np.zeros([n, vecenv.nenvs], dtype=np.bool)

  act_probs = np.zeros([n,vecenv.nenvs])


  episode_scores = []

  states[0] = vecenv.getStates()

  for i in range(n):
    policy_logits = actor.policy(vecenv.getStates())
    actions[i], act_probs[i] = actor.act(policy_logits)
    states[i+1], rewards[i], dones[i], ep_scores = vecenv.step(actions[i])
    episode_scores += ep_scores

  return (states, rewards, actions, dones, act_probs), episode_scores

''' given rollout = (states, rewards, actions, dones, act_probs)
as returned by getRollout, n = length of rollout, and actor, 
returns the GAE adv estimates and returns,
as well as the rest of the inputs needed for the batch of training
'''
def getGAE(rollout, n, actor):

  (states, rewards, actions, dones, act_probs) = rollout
  value_ests = tf.map_fn(actor.value, states)

  advs = np.zeros((n,vecenv.nenvs))

  next_adv = 0.0

  # compute returns and advs using Generalized Advantage Estimation
  for t in reversed(range(n)):
    nonterminal = (1 - dones[t])
    delta = rewards[t] + agent.DISCOUNT * value_ests[t+1] * nonterminal - value_ests[t]
    advs[t] = delta + agent.LAMBDA * agent.DISCOUNT * next_adv * nonterminal 
    next_adv = advs[t]

  states = states[:-1,...]          # toss last state and value estimate
  value_ests = value_ests[:-1,...]  

  returns = advs + value_ests

  # reshape all the collected data into a single batch before returning
  batch = [states, actions, returns, value_ests, act_probs]
  for i,x in enumerate(batch):
    batch[i] = np.reshape(x, (n*vecenv.nenvs,) + x.shape[2:])

  return batch



if __name__ == '__main__':


  parser = argparse.ArgumentParser(description="Train ppo!")
  parser.add_argument('--load_model', dest="load_model", default='', type=str, help="path to model to load")
  parser.add_argument('--savepath', dest="savepath", default='', type=str, help="path where model will be saved")
  args = parser.parse_args()

  if args.savepath:
    savepath = args.savepath
  elif args.load_model:
    savepath = args.load_model
  else:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    savedir_path = os.path.join(dir_path, 'saves')
    savepath = os.path.join(savedir_path, agent.ENVIRONMENT + str(datetime.now()))
  picklepath = os.path.join(savepath, 'save.pickle')
  
  # load model if specified, otherwise create model
  if (args.load_model):
    actor = tf.saved_model.load(savepath)
  else:
    actor = agent.Agent();
    tf.saved_model.save(actor, savepath)
    save = {}
    with open(picklepath, "wb") as fp:
      pickle.dump(save, fp)

  with open(picklepath, "rb") as f: 
    save = pickle.load(f)
    defaults = {
      'loss_policy': [], 
      'loss_value': [], 
      'loss_entropy': [],
      'episode_rewards': [],
      'framecount': 0,
      'ppo_iters': 0,
    }
    for s,v in defaults.items():
      if not s in save:
        save[s] = v

  vecenv = VecEnv(makeEnv, ENVS)
  
  cycle = 0

  for cycle in range(0, agent.FRAMES // FRAMES_PER_CYCLE):

    if cycle == 2 and PROFILE:
      tf.profiler.experimental.start('logdir')
    if cycle == 3 and PROFILE:
      tf.profiler.experimental.stop()

    rollout, episode_rewards = getRollout(vecenv, ROLLOUT_LEN, actor)
    #[states, rewards, actions, dones, act_log_probs] = rollout

    save['episode_rewards'] += episode_rewards

    if len(save['episode_rewards']) > 0:
      av_ep_rew = sum(save['episode_rewards'][-20:]) / len(save['episode_rewards'][-20:])
      print('Average episode reward: ' + str(av_ep_rew))
      print('Episode count: ' + str(len(save['episode_rewards'])))
    

    print("Frames: %d" % ((cycle + 1) * ROLLOUT_LEN * ENVS))
    print("iterations " + str(cycle + 1))
    #print("Param updates: %d" % (cycle * PARAM_UPDATES_PER_CYCLE))
  
    print('training ppo')
    total_datapoints = MINIBATCH_SIZE * MINIBATCHES
    indices = np.arange(total_datapoints)
    # TODO: does it help sample complexity to recompute GAE at each epoch? some implementations do this (minimalRL), some dont (stable baselines 3)
    # this paper https://arxiv.org/pdf/2006.05990.pdf recommends recomputing them for each epoch
    # my current implementation of getGAE is really slow
    batch = getGAE(rollout, ROLLOUT_LEN, actor)
    for e in range(EPOCHS):

      np.random.shuffle(indices)
      for mb_start in range(0,total_datapoints,MINIBATCH_SIZE):
        mb_indices = indices[mb_start:mb_start + MINIBATCH_SIZE]
        inputs = [d[mb_indices,...] for d in batch]
  
        loss_pve, grads, r, adv = actor.train(*inputs)

        loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss_pve)
  
        save['loss_policy'] += [loss_pve[0].numpy()]
        save['loss_value'] += [loss_pve[1].numpy()]
        save['loss_entropy'] += [loss_pve[2].numpy()]
        
    print(loss_str)
  
  
    cycle += 1
    if not cycle % SAVE_CYCLES:
      print('Saving model...')
      tf.saved_model.save(actor, savepath)
      with open(picklepath, "wb") as fp:
        pickle.dump(save, fp)
  
    
  
  
  
