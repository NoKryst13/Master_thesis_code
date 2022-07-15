#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

# import dmc2gym
import hydra
import gym

# from network import OFENet
import wandb


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        cfg.agent.params.env_name = cfg.env
        cfg.agent.params.aux_task = cfg.aux_task
        cfg.agent.params.total_units = cfg.total_units
        self.env = gym.make(cfg.env)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.step = 0
        self.avg_rew = 0

    @property
    def evaluate(self):
        self.agent.ofenet.state_model.eval()
        self.agent.ofenet.action_model.eval()

        average_episode_reward = 0
        count = 0
        # evaluate_buffer = ReplayBuffer(self.env.observation_space.shape,
        #                                self.env.action_space.shape,
        #                                int(self.cfg.replay_buffer_capacity),
        #                                self.device)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                state = obs
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                # evaluate_buffer.add(state, action, reward, obs, done, done)
                episode_reward += reward
                count += 1

            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        self.avg_rew = average_episode_reward
        print("---------------------------------------")
        print(f"Evaluation over {self.cfg.num_eval_episodes} episodes: {average_episode_reward:.3f}")
        print("---------------------------------------")
        self.agent.ofenet.state_model.train()
        self.agent.ofenet.action_model.train()

    def run(self):
        episode, episode_reward, done = 0, 0, True
        batch_size = 256
        episode_step = 0
        if self.cfg.wandb_name != "unnamed":
            config = {
                "env_name": self.cfg.env,
                "aux_task": self.cfg.aux_task,
                "max_timesteps": self.cfg.num_train_steps,
                "start_timesteps": self.cfg.num_seed_steps,
                "freeze_ofe": self.cfg.freeze_ofe,
                "num_pretrain": self.cfg.num_pretrain,
                "pretrain": self.cfg.pretrain,
                "total_units": self.cfg.total_units,
                "seed": self.cfg.seed,
                "batch_size": batch_size,
            }
            wandb.init(project=self.cfg.wandb_name, entity="nokryst", config=config)

        self.agent.pretrain_ofe(self.env, self.replay_buffer, random_collect=self.cfg.num_pretrain, batch_size=batch_size)

        self.step = self.cfg.num_pretrain

        eval_flag = False
        while self.step < self.cfg.num_train_steps:
            if self.cfg.wandb_name != "unnamed":
                if self.step % 500 == 0:
                    wandb_logs = {
                        "Reward": self.avg_rew,
                        "Step": self.step,
                        "train": {
                            "mse": self.agent.ofenet.mse_train,

                        },
                        "test": {
                            "mae": self.agent.ofenet.mae_test,
                            "mae_percent": self.agent.ofenet.mae_percent_test,
                            "mape": self.agent.ofenet.mape_test,
                            "mse": self.agent.ofenet.mse_test,
                            "mse_percent": self.agent.ofenet.mse_percent_test,
                        }
                    }
                    wandb.log(wandb_logs)

            if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                eval_flag = True

            if done:
                # evaluate agent periodically
                if eval_flag:

                    self.evaluate
                    eval_flag = False
                print(
                    f"Total T: {self.step + 1} Episode Num: {episode + 1} Episode T: {episode_step} Reward: {episode_reward:.3f}")
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            # collecting samples is not needed, if ofenet was pretrained.
            if self.step >= self.cfg.num_seed_steps:
                sample_state, sample_action, sample_reward, sample_next_state, sample_not_done, \
                    sample_not_dones_no_max = self.replay_buffer.sample(batch_size=batch_size)
                self.agent.ofenet.train_ofe(sample_state, sample_action, sample_next_state, sample_reward, sample_not_done)

                self.agent.update(self.replay_buffer, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
