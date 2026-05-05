from env.env_combat import AircraftEnv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
import time

config = {"mission_config": "cannon/L1_cannon", "hierarchy_level":0, "training_level": 1}
# config = {"mission_config": "track/L1_track", "hierarchy_level":0, "training_level": 1}

env = AircraftEnv(config)
obs, _ = env.reset()

# act = {"B1": [15, 13.7, 15, 3, 1]}
act = {"B1": [5, 5, 5, 5, 1]}
done = False
steps = 0



# s,r,t,tt,_ = env.step(act)
# # print("reward -- ", r)
# while True:
#     env.render(viewpoint=0, trajectory_view=False)



while not done:
    #print(f"---------- iteration {steps} ------------")
    env.render(viewpoint=0, trajectory_view=False)
    s,r,t,tt,_ = env.step(act)
    steps += 1
    # done = t["__all__"] or tt["__all__"]
    done = steps==5




# while not done:
#     env.render(viewpoint=0, trajectory_view=False)
#     # time.sleep(0.4)
#     if STATIC_ACT:
#         s,r,t,tt,_ = env.step(act)
#     else:
#         act = env.action_space.sample()
#         s,r,t,tt,_ = env.step(act)

#     #print(f"step {steps} -- obs shoot ", s["B1"][13:15])
#     print("rew ", r)
#     done = t["__all__"] or tt["__all__"]
#     steps += 1
# # while done:
# #     env.render(viewpoint=2, trajectory_view=False)
# print(f"steps {steps}")