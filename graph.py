import matplotlib.pyplot as plt
import argparse
import os
import pickle
import numpy as np

import agent

parser = argparse.ArgumentParser(description="Graph ppo!")
parser.add_argument('load_model', type=str, help="path to model to load")
args = parser.parse_args()

savepath = args.load_model
picklepath = os.path.join(savepath, 'save.pickle')

with open(picklepath, "rb") as f: 
  save = pickle.load(f)


fig, [ax1,ax2,ax3,ax4] = plt.subplots(nrows=4)

ax1.set_title('Unsmoothed training graphs')

ax1.plot(save['loss_policy'])
ax1.set_ylabel('Policy loss')
ax2.plot(save['loss_value'])
ax2.set_ylabel('Value loss')
ax3.plot(save['loss_entropy'])
ax3.set_ylabel('Entropy loss')
ax3.set_xlabel('PPO training epochs')
ax4.plot(save['episode_rewards'])
ax4.set_ylabel('Cumulative episode rewards (binned to -1,0,1)')
ax4.set_xlabel('Episode number')

#ax.grid(True)
#start, end = ax.get_ylim()
#ax.set_ylim(0,0.1)
#ax.yaxis.set_ticks(np.arange(0, 0.1, 0.001))
#ax.set_yscale('log')
plt.show()
