import gym, os
from stable_baselines3 import A2C, PPO

from eater import EaterEnv

# Directories
models_dir = 'models/PPO'
logs_dir = 'logs'

# Env
env = EaterEnv()
env.reset()

# Train
model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=logs_dir)
TIMESTAMPS = 10_000
for iterations in range(1, 100):
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f'{models_dir}/{TIMESTAMPS * iterations}')