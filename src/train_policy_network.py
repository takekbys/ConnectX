from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import random
from torchsummary import summary
import itertools

from helper import ConnectXEnv
from helper import CustomCNN
from helper import Callback
from helper import make_ConnectXEnv
from helper import RolloutAgent
from helper import RolloutAgent

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

DEBUG = True
NUM_ENV = 8
N_STEPS = 512
LOG_DIR = "./logs"
MODEL_DIR = "./models"
TRAIN_TIMESTEPS = 10000
TRAIN_SET = 1000
MODEL_SAVE_FREQ = 10

def testRollout():

    from helper import RolloutAgent
    from kaggle_environments import make
    from kaggle_environments.envs.connectx.connectx import renderer
 
    # prepare rollout opponent agent
    model_name = "/tmp"
    modelpath = MODEL_DIR + model_name
    rollout = RolloutAgent(modelpath)

    print("Test model")
    env = make("connectx", debug=False)
    trainer = env.train([None, rollout])
    state = trainer.reset()

    done = False
    while not done:
        state, reward, done, info = trainer.step(0)
    print("reward : ", reward)
    print(renderer(env.state, env))

def train_policy_network():
    # 自分のモデルの読み込み
    start_model_name = "/tmp"
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=7)
    )
    #model = PPO("CnnPolicy", train_envs, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=LOG_DIR)

    # 対戦相手のモデルリストを作成
    opponent_list = []
    model_name = "/tmp"
    opponent_list.append(MODEL_DIR+model_name)

    # for 規定のセット回数
    for i in range(TRAIN_SET):

        # 対戦相手をランダムに選択
        opponent_path = random.choice(opponent_list[max(0,len(opponent_list)-10):])
        rollout = RolloutAgent(opponent_path)

        # 対戦環境を構築
        turns = ["sente", "gote"]
        train_envs = DummyVecEnv([make_ConnectXEnv(i, policy=rollout, turn=turns[i%2]) for i in range(NUM_ENV)])
        model = PPO.load(MODEL_DIR+model_name, env=train_envs, n_steps=N_STEPS, tensorboard_log=None)

        if DEBUG and i==0:
            print("---------------------------------------")
            print("Observation space : ", model.policy.observation_space)
            print("Summary of feature extraction layers : ")
            print(summary(model.policy.features_extractor, model.policy.observation_space.shape))

        # train
        # in case of alpha Go, model trains 128 games per each train set.
        # For ConnectX, what number is appropriate? 128試合x縦6*横7=5376手
        model.learn(total_timesteps=TRAIN_TIMESTEPS)

        # evaluate and save model
        # (model vs random), (model vs negamax), (random vs model), (negamax vs model)
        if (i%MODEL_SAVE_FREQ)==0:
            # evaluate
            policies = ["random", "negamax"]
            env_settings = itertools.product(policies, turns)

            mean_rewards = []
            for p in env_settings:
                val_env = DummyVecEnv([make_ConnectXEnv(0, policy=p[0], turn=p[1])])
                mean_rewards.append(validate(model, val_env))
            reward = np.mean(mean_rewards)
            print("Train set : ", i)
            print("Validation mean reward : ", np.mean(mean_rewards))

            # save
            model_name = "/model{:05}_eval{:.3f}".format(i, reward)
            opponent_list.append(MODEL_DIR+model_name)
            model.save(MODEL_DIR + model_name)

            # output 



if __name__=="__main__":
    train_policy_network()