import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt 
import math

import networkx as nx

np.random.seed(100)

class PolicyNN(nn.Module):
  def __init__(self, input_layer_size, hidden_layer_sizes, output_layer_size):
    super(PolicyNN, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_layer_size, hidden_layer_sizes[0]),
      nn.ReLU(),
      nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
      nn.ReLU(),
      nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
      nn.ReLU(),
      nn.Linear(hidden_layer_sizes[2], output_layer_size)
    )

  def forward(self, x):
    x = self.layers(x)
    return x

VERTICES = 19
NUMBER_OF_POSSIBLE_EDGES = (VERTICES * (VERTICES - 1)) // 2

OBSERVATION_SPACE_SIZE = NUMBER_OF_POSSIBLE_EDGES
ACTION_SPACE_SIZE = NUMBER_OF_POSSIBLE_EDGES

NUMBER_OF_ACTIONS = 2 # add or don't add edge

INPUT_LAYER_SIZE = OBSERVATION_SPACE_SIZE + ACTION_SPACE_SIZE
HIDDEN_LAYER_SIZES = [128, 64, 4] # arbitrary
OUTPUT_LAYER_SIZE = NUMBER_OF_ACTIONS

policyNet = PolicyNN(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZES, OUTPUT_LAYER_SIZE)

def construct_graph(state):
  G = nx.Graph()
  G.add_nodes_from(list(range(VERTICES)))
  count = 0

  for i in range(VERTICES):
    for j in range(i+1,VERTICES):
      if state[count] == 1:
        G.add_edge(i,j)
        G.add_edge(j,i)
      count += 1
  
  return G

def check_connectivity(G):
  return nx.is_connected(G)

def calculate_eigenvalues(G):
  evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
  evalsRealAbs = np.zeros_like(evals)

  for i in range(len(evals)):
    evalsRealAbs[i] = abs(evals[i])

  lambda1 = max(evalsRealAbs)
  
  return lambda1

def calculate_matching_number(G):
  maxMatch = nx.max_weight_matching(G)
  mu = len(maxMatch)
  
  return mu

def calculate_score(G, state, lambda1, mu):
  myScore = math.sqrt(VERTICES - 1) + 1 - lambda1 - mu

  if myScore > 0:
    # Counterexample found. Print state and draw graph.
    print(state)
    nx.draw_kamada_kawai(G)
    plt.show()
    exit()
      
  return myScore

def calculate_reward(state):
  G = construct_graph(state)
  
  # If the graph is not connected, return a very negative reward.
  if not check_connectivity(G):
      return -1e9

  lambda1 = calculate_eigenvalues(G)
  mu = calculate_matching_number(G)
  myScore = calculate_score(G, state, lambda1, mu)

  return myScore

from collections import namedtuple

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action', 'action_representation'])

def select_action(state, step_count, net):
  action_representation = np.zeros(len(state))
  action_representation[step_count] = 1
  
  sm = nn.Softmax(dim=0)

  input = torch.FloatTensor([*state, *action_representation])

  output = net(input)

  # print(output)

  action_probabilities_v = sm(output)

  # print(action_probabilities_v)

  action_probabilities = action_probabilities_v.data.numpy()
        
  action = np.random.choice(len(action_probabilities), p=action_probabilities)

  return action

import gym
from gym.spaces import Graph, Box, Discrete

class GraphGameEnv(gym.Env):
  def __init__(self):
    super(GraphGameEnv, self).__init__()

    self.observation_space = Discrete(OBSERVATION_SPACE_SIZE)
    self.action_space = Discrete(ACTION_SPACE_SIZE)

    self.current_state = None
    self.step_count = 0

  def step(self, action):
    # Calculate the position from the step count
    position = self.step_count

    # If there's no edge at the given position, add an edge
    if action == 1 and self.current_state[position] == 0:
        self.current_state[position] = 1

    self.step_count += 1

    # Check if the episode is over
    terminal = self.step_count == NUMBER_OF_POSSIBLE_EDGES

    if terminal:
      reward = calculate_reward(self.current_state)
      done = True
    else:
      reward = 0
      done = False

    return self.current_state, reward, done, {}

  def reset(self):
    # At the start of an episode, there are no edges in the graph
    self.current_state = np.zeros(NUMBER_OF_POSSIBLE_EDGES)
    self.step_count = 0
    return self.current_state

  def render(self, mode='human'):
    G = construct_graph(self.current_state)
    nx.draw_kamada_kawai(G)
    plt.show()

  def close(self):
    pass

def iterate_batches(env, net, batch_size):
  batch = []
  episode_reward = 0.0
  episode_steps = []
  obs = env.reset()
  
  while True:
    action = select_action(obs, env.step_count, net);

    action_representation = np.zeros(len(obs))
    action_representation[env.step_count] = 1

    next_obs, reward, is_done, _ = env.step(action)

    episode_reward += reward

    step = EpisodeStep(observation=obs, action=action, action_representation=action_representation)
    episode_steps.append(step)

    if is_done:
      e = Episode(reward=episode_reward, steps=episode_steps)
      batch.append(e)

      episode_reward = 0.0
      episode_steps = []
      
      next_obs = env.reset()

      if len(batch) == batch_size:
        yield batch
        batch = []

    obs = next_obs

def filter_batch(batch, percentile):
  rewards = list(map(lambda s: s.reward, batch))
  reward_bound = np.percentile(rewards, percentile)
  reward_mean = float(np.mean(rewards))

  train_obs = []
  train_act = []
  train_act_rep = []

  for episode in batch:
    if episode.reward < reward_bound:
      continue
    train_obs.extend(map(lambda step: step.observation, episode.steps))
    train_act_rep.extend(map(lambda step: step.action_representation, episode.steps))
    train_act.extend(map(lambda step: step.action, episode.steps))

  train_obs_v = torch.FloatTensor(train_obs)
  train_act_rep_v = torch.FloatTensor(train_act_rep)
  train_act_v = torch.LongTensor(train_act)
  
  return train_obs_v, train_act_v, train_act_rep_v, reward_bound, reward_mean

BATCH_SIZE = 1000
PERCENTILE = 93

env = GraphGameEnv()

# env = gym.wrappers.Monitor(env, directory="mon", force=True)

# obs_size = env.observation_space.shape[0]
# n_actions = env.action_space.ny

obs_size = NUMBER_OF_POSSIBLE_EDGES
OBSERVATION_SPACE_SIZE = NUMBER_OF_POSSIBLE_EDGES
ACTION_SPACE_SIZE = NUMBER_OF_POSSIBLE_EDGES

n_actions = 2 # add or don't add edge

INPUT_LAYER_SIZE = OBSERVATION_SPACE_SIZE + ACTION_SPACE_SIZE
HIDDEN_LAYER_SIZES = [128, 64, 4] # arbitrary
OUTPUT_LAYER_SIZE = NUMBER_OF_ACTIONS

policyNet = PolicyNN(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZES, OUTPUT_LAYER_SIZE)
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=policyNet.parameters(), lr=0.0001)
writer = SummaryWriter(comment="-graph")

for iter_no, batch in enumerate(iterate_batches(env, policyNet, BATCH_SIZE)):
    obs_v, acts_v, train_act_rep_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
    optimizer.zero_grad()

    input = torch.cat((obs_v, train_act_rep_v), -1)

    action_scores_v = policyNet(input)

    acts_v_one_hot = F.one_hot(acts_v, n_actions).float()

    loss_v = objective(action_scores_v, acts_v_one_hot)
    loss_v.backward()
    optimizer.step()
    print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
        iter_no, loss_v.item(), reward_m, reward_b))
    writer.add_scalar("loss", loss_v.item(), iter_no)
    writer.add_scalar("reward_bound", reward_b, iter_no)
    writer.add_scalar("reward_mean", reward_m, iter_no)
    if reward_m > 199:
        print("Solved!")
        break
writer.close()