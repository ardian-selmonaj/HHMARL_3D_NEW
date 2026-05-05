from env.env_base import AircraftEnv


config = {"mission_config":"cannon/L1_cannon", "hierarchy_level":0, "training_level": 1}
env = AircraftEnv(config)


obs, _ = env.reset()

print("action space ", env.action_space)
print("action size ", env.action_space["B1"].shape[0])
print("obs space ", env.observation_space)