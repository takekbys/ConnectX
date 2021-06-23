from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import random
from torchsummary import summary
import itertools
import gc

from helper import ConnectXEnv
from helper import CustomCNN
from helper import Callback
from helper import make_ConnectXEnv
from helper import RolloutAgent

def train_policy_network(
    num_env=1,
    env_type="DummyVecEnv",
    n_steps=1,
    train_timesteps=1,
    train_set=1,
    train_set_start=0,
    model_save_freq=1,
    start_model=None,
    opponent_list=[],
    need_eval=False,
    log_dir="./logs",
    model_dir="./models",
    debug=False
    ):
    
    # setup
    if env_type=="DummyVecEnv":
        env_wrapper = DummyVecEnv
    elif env_type=="SubprocVecEnv":
        env_wrapper = SubprocVecEnv

    model_name = start_model

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=7)
    )

    # create opponent model list. duaring training, this list is updated
    if opponent_list==[]:
        model_name = "tmp"
        opponent_list.append(model_dir + "/" + model_name)
    else:
        for i, name in enumerate(opponent_list):
            opponent_list[i] = model_dir + "/" + name


    for i in range(train_set_start, train_set):

        # select opponent model randomly
        opponent_path = random.choice(opponent_list[max(0,len(opponent_list)-10):])
        rollout = RolloutAgent(opponent_path)

        # build train environments
        turns = ["sente", "gote"]
        train_envs = env_wrapper([make_ConnectXEnv(i, policy=rollout, turn=turns[i%2]) for i in range(num_env)])
        # if start_model is not given, initialize randomly
        if start_model==None and i==0:
            model = PPO('CnnPolicy', env=train_envs, n_steps=n_steps, policy_kwargs=policy_kwargs, verbose=1)
        # if start_model is given, use saved model
        else:
            tmp = model_dir + "/" + model_name
            model = PPO.load(tmp, env=train_envs, n_steps=n_steps, tensorboard_log=None)

        if DEBUG and i==0:
            print("---------------------------------------")
            print("Observation space : ", model.policy.observation_space)
            print("Summary of feature extraction layers : ")
            print(summary(model.policy.features_extractor, model.policy.observation_space.shape))

        # train
        # in case of alpha Go, model trains 128 games per each train set.
        # For ConnectX, what number is appropriate? 128試合x縦6*横7=5376手
        print("Train set : ", i)
        print("Model : ", model_name.split("/")[-1], ", Opponent : ", opponent_path.split("/")[-1])
        model.learn(total_timesteps=train_timesteps)

        # evaluate and save model
        # (model vs random), (model vs negamax), (random vs model), (negamax vs model)
        if (i%MODEL_SAVE_FREQ)==0 and i>0:
            # evaluate
            if need_eval:
                policies = ["random", "negamax"]
                env_settings = itertools.product(policies, turns)

                mean_rewards = []
                for p in env_settings:
                    val_envs = env_wrapper([make_ConnectXEnv(0, policy=p[0], turn=p[1])])
                    mean_reward, _ = evaluate_policy(model, val_envs, min(20,num_env))
                    mean_rewards.append(mean_reward)
                reward = np.mean(mean_rewards)
                print("Validation mean reward : ", reward)

                # delete gabage
                del val_envs
            
            # save model
            if need_eval:
                model_name = "model{:05}_eval{:.3f}".format(i, reward)
            else:
                model_name = "model{:05}".format(i)
            opponent_list.append(model_dir + "/" + model_name)
            model.save(model_dir + "/" + model_name)
        
        else:
            # save current model in order to load in next train set
            model_name = "model"
            model.save(model_dir + "/" + model_name)

        # delete gabage
        del model, train_envs
        gc.collect()

            # output 

DEBUG = True
NUM_ENV = 8
N_STEPS = 128
TRAIN_TIMESTEPS = 1000
TRAIN_SET = 1000
MODEL_SAVE_FREQ = 10

if __name__=="__main__":
    train_policy_network(
    num_env=NUM_ENV,
    env_type="DummyVecEnv",
    n_steps=N_STEPS,
    train_timesteps=TRAIN_TIMESTEPS,
    train_set=TRAIN_SET,
    train_set_start=10,
    model_save_freq=MODEL_SAVE_FREQ,
    start_model="model",
    opponent_list=[],
    need_eval=False,
    debug=DEBUG
    )