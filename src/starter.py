from kaggle_environments import make, utils
from kaggle_environments.envs.connectx.connectx import renderer
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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

## create env
env = make("connectx", debug=False)
#env = ConnectXEnv(env, policy="negamax")
connectXEnv = ConnectXEnv(env, policy="negamax")

print("checking env...")
check_env(connectXEnv, True, True)
print("check done.")

BLEnv = DummyVecEnv([lambda: connectXEnv])

## define callback

class TqdmCallback(object):
    def __init__(self):
        self.pbar = None
    
    def __call__(self, _locals, _globals):
        if self.pbar is None:
            self.pbar = tqdm(total=_locals['nupdates'])
            
        self.pbar.update(1)
        
        if _locals['update'] == _locals['nupdates']:
            self.pbar.close()
            self.pbar = None

        return True

callback = TqdmCallback()

## train agent of ConnectX
model = PPO("MlpPolicy", BLEnv, verbose=0)
model.learn(total_timesteps=100000)

print("train done")

## play ConnectX
state = connectXEnv.reset()
while not connectXEnv.done:
    action, _states = model.predict(state, deterministic=True)
    state, reward, done, info = connectXEnv.step(action)
    print(f"reward: {reward}, done: {done}, info: {info}")
    connectXEnv.renderBoard()