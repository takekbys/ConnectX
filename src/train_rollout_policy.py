
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import random

from helper import ConnectXEnv
from helper import CustomCNN
from helper import Callback
from helper import make_ConnectXEnv


def train(model, envs, total_timesteps=100):
    callback = Callback()
    model.learn(total_timesteps=total_timesteps)#, callback=callback)

    print("train done")

    return model

def validate(model, env, num_episodes=100):
    all_episode_rewards = [] # list of rewards on episodes
    for i in range(num_episodes):
        episode_rewards = 0 # reward on a episode
        done = False
        state = env.reset()
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)
            if done:
                all_episode_rewards.append(reward)
    return np.mean(all_episode_rewards)


def test(model, env, turn):
    ## play ConnectX
    state = env.reset()
    while not env.done:
        action, _states = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        print(f"reward: {reward}, done: {done}, info: {info}")
    env.renderBoard()


NUM_ENV = 8
LOG_DIR = "./logs"
MODEL_DIR = "./models"
TRAIN_TIMESTEPS = 100000

def main():
    # train
    turns = ["sente", "gote"]
    train_envs = DummyVecEnv([make_ConnectXEnv(i, policy="random", turn=turns[i%2]) for i in range(NUM_ENV)])
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=7)
    )
    model = PPO("CnnPolicy", train_envs, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=LOG_DIR)
    
    model = train(model, train_envs, total_timesteps=TRAIN_TIMESTEPS)
    model_name = "/tmp"
    model.save(MODEL_DIR + model_name)

    # validate
    model = PPO.load(MODEL_DIR + model_name)
    turns = ["sente", "gote"]
    mean_rewards = []
    for turn in turns:
        val_env = DummyVecEnv([make_ConnectXEnv(0, policy="random", turn=turn)])
        mean_rewards.append(validate(model, val_env))
    print("validation mean reward : ", np.mean(mean_rewards))

    # test
    model = PPO.load(MODEL_DIR + model_name)
    print("Test model")
    for turn in ["sente", "gote"]:
        print("Turn is set as \"" + turn + "\"")
        env = ConnectXEnv(policy="random", turn=turn)
        test(model, env, turn)

    env.boardshow()

if __name__=="__main__":
    main()