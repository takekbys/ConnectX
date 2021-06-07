from kaggle_environments import make, utils
from kaggle_environments.envs.connectx.connectx import renderer
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import datetime
import pytz

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor

## define env for connectx trainig
class ConnectXEnv(gym.Env):
    
    def __init__(self, env, policy="negamax"):
        super(ConnectXEnv, self).__init__()
        self.env = env
        self.trainer = self.env.train([None, policy])

        # アクション数定義
        ACTION_NUM = env.configuration["columns"]
        self.action_space = gym.spaces.Discrete(ACTION_NUM)

        # 状態の範囲を定義
        self.WIDTH = env.configuration["columns"]
        self.HEIGHT = env.configuration["rows"]
        
        self.obs_shape = [self.HEIGHT, self.WIDTH, 3] # treat as image
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        
        self.num_envs = 1
        self.reset()

    def reset(self):
        state = self.trainer.reset()
        self.state = state
        self.board = state["board"]
        self.mark = state["mark"]
        self.done = False
        self.boardImg = self.board2img(self.board)
        
        return self.boardImg

    def step(self, action):
        action = int(action) # added this row in order to avoid error like "Invalid Action: 4 is not of type 'integer'"
        state, reward, done, info = self.trainer.step(action)
        
        self.state = state
        self.board = state["board"]
        self.boardImg = self.board2img(self.board)
        self.reward = -1 if reward==None else reward
        self.done = done

        return self.boardImg, self.reward, self.done, info

    def render(self, **kwargs):
        self.env.render(**kwargs)
        
    def board2img(self, board):
        tmp = np.array(board).reshape(self.obs_shape[:2])
        boardImg = np.zeros(self.obs_shape, dtype=np.uint8)
        boardImg[:,:,0] = np.array(tmp==self.mark)*255 # ch.0 : own mark
        boardImg[:,:,1] = np.array(tmp==(3-self.mark))*255 # ch.1 : oponent mark
        return boardImg
    
    def boardshow(self, verbose=0):
        plt.figure(figsize=(2,2))
        plt.imshow(self.boardImg)
        if verbose:
            print("red : own piece")
            print("green : oponent piece")
    
    def renderBoard(self):
        columns = self.env.configuration.columns
        rows = self.env.configuration.rows
        board = self.state.board

        def print_row(values, delim="|"):
            return f"{delim} " + f" {delim} ".join(str(v) for v in values) + f" {delim}\n"

        row_bar = "+" + "+".join(["---"] * columns) + "+\n"
        out = row_bar
        for r in range(rows):
            out = out + \
                print_row(board[r * columns: r * columns + columns]) + row_bar

        print(out)

## train agent of ConnectX
# define custom CNN
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1), # 3x6x7 to 32x5x6
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 32x5x6 to 64x4x5
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 64x4x5 to 128x3x34
            nn.ReLU(),
            nn.Flatten(), # 128x3x4 to 1536
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=7),
)

# define callback
num_update = 0
best_mean_reward = -np.inf

def callback(_locals, _globals):
    global num_update
    global best_mean_reward

    if (num_update+1)%1000==0:
        _, y = ts2xy(load_results(model_dir), "timesteps")
        if len(y)>0:
            mean_reward = np.mean(y[-1000:])
            update_model = mean_reward > best_mean_reward
            if update_model:
                best_mean_reward = mean_reward
                _locals["self"].save("./models/best_model")

            print("time: {}, num_update: {}, mean: {:.3f}, best_mean: {:.3f}, model_update: {}".format(
                datetime.datetime.now(pytz.timezone("Asia/Tokyo")),
                num_update,
                mean_reward,
                best_mean_reward,
                update_model
            ))

    num_update += 1
    return True


## create env
env = make("connectx", debug=False)
connectXEnv = ConnectXEnv(env, policy="random")

# print("checking env...")
# check_env(connectXEnv, True, True)
# print("check done.")

model_dir = "./models"
BLEnv = Monitor(connectXEnv, model_dir, allow_early_resets=True)
BLEnv = DummyVecEnv([lambda: BLEnv])

# train
log_dir = "./logs"
model = PPO("CnnPolicy", BLEnv, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=1000000, callback=callback)

print("train done")

## play ConnectX
state = connectXEnv.reset()
while not connectXEnv.done:
    action, _states = model.predict(state, deterministic=True)
    state, reward, done, info = connectXEnv.step(action)
    print(f"reward: {reward}, done: {done}, info: {info}")
    connectXEnv.renderBoard()