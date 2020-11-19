## SpaceInvaders imports
import gym
from torch_deep_q_model import DeepQNetwork, Agent
from utils import plotLearning, get_optimizer, exploit_and_explore
import numpy as np
from gym import wrappers
## Population Based Learning imports
import argparse
import os
import pathlib
import torch
import torch.nn as nn
import torch.multiprocessing as _mp
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from model import DeepQNetwork, Agent


mp = _mp.get_context('spawn')

class Worker(mp.Process):
    def __init__(self, batch_size, epoch, max_epoch, train_data, test_data, population, finish_tasks,
                 device):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.device = device
        model = DeepQNetwork().to(device)
        optimizer = get_optimizer(model)
        self.trainer = Agent(model=model,
                               optimizer=optimizer,
                               loss_fn=nn.CrossEntropyLoss(),
                               train_data=train_data,
                               test_data=test_data,
                               batch_size=self.batch_size,
                               device=self.device)

    def run(self):
        while True:
            if self.epoch.value > self.max_epoch:
                break
            # Train
            task = self.population.get()
            self.trainer.set_id(task['id'])
            checkpoint_path = "checkpoints/task-%03d.pth" % task['id']
            if os.path.isfile(checkpoint_path):
                self.trainer.load_checkpoint(checkpoint_path)
            try:
                self.trainer.train()
                score = self.trainer.eval()
                self.trainer.save_checkpoint(checkpoint_path)
                self.finish_tasks.put(dict(id=task['id'], score=score))
            except KeyboardInterrupt:
                break


class Explorer(mp.Process):
    def __init__(self, epoch, max_epoch, population, finish_tasks, hyper_params):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.hyper_params = hyper_params

    def run(self):
        while True:
            if self.epoch.value > self.max_epoch:
                break
            if self.population.empty() and self.finish_tasks.full():
                print("Exploit and explore")
                tasks = []
                while not self.finish_tasks.empty():
                    tasks.append(self.finish_tasks.get())
                tasks = sorted(tasks, key=lambda x: x['score'], reverse=True)
                print('Best score on', tasks[0]['id'], 'is', tasks[0]['score'])
                print('Worst score on', tasks[-1]['id'], 'is', tasks[-1]['score'])
                fraction = 0.2
                cutoff = int(np.ceil(fraction * len(tasks)))
                tops = tasks[:cutoff]
                bottoms = tasks[len(tasks) - cutoff:]
                for bottom in bottoms:
                    top = np.random.choice(tops)
                    top_checkpoint_path = "checkpoints/task-%03d.pth" % top['id']
                    bot_checkpoint_path = "checkpoints/task-%03d.pth" % bottom['id']
                    exploit_and_explore(top_checkpoint_path, bot_checkpoint_path, self.hyper_params)
                    with self.epoch.get_lock():
                        self.epoch.value += 1
                for task in tasks:
                    self.population.put(task)



if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    brain = Agent(gamma=0.95, epsilon=1.0,
                  alpha=0.003, maxMemorySize=5000,
                  replace=None)
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            if done and info['ale.lives'] == 0:
                reward = -100
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward,
                                np.mean(observation_[15:200,30:125], axis=2))
            observation = observation_
    print('done initializing memory')

    scores = []
    epsHistory = []
    numGames = 50
    batch_size=32
    # uncomment the line below to record every episode.
    env = wrappers.Monitor(env, "tmp/space-invaders-1", video_callable=lambda episode_id: True, force=True)
    for i in range(numGames):
        print('starting game ', i+1, 'epsilon: %.4f' % brain.EPSILON)
        epsHistory.append(brain.EPSILON)
        done = False
        observation = env.reset()
        frames = [np.sum(observation[15:200,30:125], axis=2)]
        score = 0
        lastAction = 0
        while not done:
            if len(frames) == 3:
                action = brain.chooseAction(frames)
                frames = []
            else:
                action = lastAction
            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(observation_[15:200,30:125], axis=2))
            if done and info['ale.lives'] == 0:
                reward = -100
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward,
                                  np.mean(observation_[15:200,30:125], axis=2))
            observation = observation_
            brain.learn(batch_size)
            lastAction = action
            #env.render(
        scores.append(score)
        print('score:',score)
    x = [i+1 for i in range(numGames)]
    fileName = str(numGames) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
               'Alpha' + str(brain.ALPHA) + 'Memory' + str(brain.memSize)+ '.png'
    plotLearning(x, scores, epsHistory, fileName)
