from stable_baselines3 import PPO, A2C
from eater import EaterEnv

# Env
env = EaterEnv()
env.reset()

# Train
models_dir = 'models/PPO'
model_path = f'{models_dir}/10000.zip'
model = PPO.load(model_path, env=env)

episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()

env.close()