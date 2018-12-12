# UnityML Collaboration and Competition Project

This project demonstrates how two reinforcement learning agents can learn to play tennis cooperatively.

## Requirements

* Windows (64 bit)
* [Python 3.6](https://www.python.org/downloads/release/python-366/)
* [Unity ML-Agents Toolkit](https://www.python.org/downloads/release/python-366/)
* [Pytorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/) 
* [Jupyter](http://jupyter.org/) 

## Installation
Recommended way of installing the dependencies is via [Anaconda](https://www.anaconda.com/download/). To create a new Python 3.6 environment run

`conda create --name myenv python=3.6`

Activate the environment with

`conda activate myenv`

[Click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) for instructions on how to install the Unity ML-Agents Toolkit.

Visit [pytorch.org](https://pytorch.org/) for instructions on installing pytorch.

Install matplotlib with

`conda install -c conda-forge matplotlib`

Jupyter should come installed with Anaconda. If not, [click here](http://jupyter.org/install) for instructions on how to install Jupyter.


## Getting started
The project can be run with the provided jupyter notebooks. Tennis_Observation.ipynb allows one to observe a fully trained agents in the environment. Tennis_Training.ipynb can be used to train a new agents or continue training pre-trained agents. Several pre-trained agents are stored in the `savedata` folder. 

## Environment
The environment is a a tennis field with two racks and a net. Each rack has its own observation and action space. The observation space consists of 24 continuous variables (8 variables stacked over three steps), e.g. position from the net. The action space is continuous and two dimensional. There is one action for moving towards the net and away from it, and one for jumping. Each agent receives a positive reward when it manages to push the ball over the net and a negative reward when it lets the ball drop to the floor.

## Algorithm
For this project [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) are used in a [multi-agent setting](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). A detailed description of DDPG and MADDPG can be found in the linked papers.

## Agents
The model uses an actor-critic approach. This is realized by fully connected feed forward artificial neural networks with two hidden layers at 128 units each with ReLU activation. Each agent has an actor network that takes the agent's observation and outputs a two-dimensional action vector. The actions and observations of both agents are then used as input for the shared critic network. All networks use local copies during optimization and then perform soft-updates on their target networks. Both agents also use a shared memory buffer to store experiences and sample values for training.

## Training
During training, the agents observe states, predict actions and then observe resulting rewards and follow-up states. The critic then tries to predict the value of each action per state and the actors are trained using gradient descent to maximize this value. After each learning step, the target networks are updated using soft-update. The following parameters were used for training.

| parameter   | value    |  description |
|---------|---------------|-------------|
|BUFFER_SIZE| 100000| replay buffer size |
BATCH_SIZE | 128        | minibatch size|
GAMMA | 0.99            | discount factor|
TAU | 0.002              | for soft update of target parameters |
LR_ACTOR | 3e-4         | learning rate of the actor |
LR_CRITIC | 1e-4        | learning rate of the critic|
WEIGHT_DECAY | 0        | L2 weight decay - set to 0 to prevent rewards from drowning |

Training can be performed on cpu or gpu. The default is cpu and the setting is stored in the `device` variable of agents.py.

With these settings, the agent should learn to solve the environment 
in approximately 2200 episodes.


