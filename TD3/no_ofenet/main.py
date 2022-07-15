import numpy as np
import torch
import gym
import argparse
import os
import sys

import utils
import TD3
import time
from progress.bar import Bar

import wandb


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):

    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    counter = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:
            obs = state
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            counter += 1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=10e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=10e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic, old=100, new=256
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--aux_task", default="no")
    parser.add_argument("--wandb_name", default="off")
    parser.add_argument("--learning_rate", default="3e-4", type=float) # old = 1e-3, new = 3e-4
    args = parser.parse_args()

    env = gym.make(args.env)
    dummy_env = gym.make(args.env)
    dummy_env.reset()

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    state, done = env.reset(), False

    ###

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    ##
    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action, "discount": args.discount,
              "tau": args.tau, "policy_noise": args.policy_noise * max_action,
              "noise_clip": args.noise_clip * max_action, "policy_freq": args.policy_freq, "learning_rate": args.learning_rate}

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    policy = TD3.TD3(**kwargs)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_return = 0
    avg_rew = 0

    if args.wandb_name != "off":
        config = {
            "env_name": args.env,
            "aux_task": args.aux_task,
            "max_timesteps": args.max_timesteps,
            "start_timesteps": args.start_timesteps,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }

        wandb.init(project=args.wandb_name, entity="nokryst", config=config)
    for t in range(int(args.max_timesteps)):
        start_time = time.time()


        if args.wandb_name != "off":
            if t % 500 == 0:
                wandb_logs = {
                    "Reward": avg_rew,
                    "Step": t}
                wandb.log(wandb_logs)

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action

        next_state, reward, done, _ = env.step(action)

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_rew = eval_policy(policy, args.env, args.seed)
            evaluations.append(avg_rew)

