This is a simple implementation of PPO.

All hyperparams and settings are global variables in the fiels (agent.py and ppo.py).
To run on different environments, change the ENVIRONMENT variable in agent.py
and (any corresponding hyperparameters) and then run 
python ppo.py


Below is a gif of the agent playing space invaders!

![one episode of space invaders](images/space_invaders_animation.gif)

And this is a graph of the unsmoothed training curves for 40M frames of training on space invaders:
![space invaders training curves](images/space_invaders_graph.png)

