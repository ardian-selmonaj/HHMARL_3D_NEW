from rewards.reward_base import BaseRewardFunction
import math


class DistanceReward(BaseRewardFunction):
    """
    Higher reward the further away from opponent. 
    Designed as potential reward.
    """
    def __init__(self, config):
        super().__init__(config)
        self.map_size = getattr(self.config, 'map_size', 50) # map size in km
        self.half_map = self.map_size / 2

    def get_reward(self, task, env, agent_id, action=None):
        #reward = self.dist_reward(env, agent_id)
        delta_x, delta_y, _, _, _, _ = super().get_delta_values(env, agent_id, in_km=True)
        scale, shift = task.shift_range(0, 50, -5, 10)
        x = delta_x * scale + shift
        y = delta_y * scale + shift
        reward = 0.5*x + 0.5*y
        return self._process(reward, agent_id)


    # def dist_reward(self, env, agent_id):

    #     def delta_loc(x):
    #         # in [KM]
    #         dist_scale = 10 #7
    #         dist_offset = 2 #5
    #         if x < dist_offset:
    #             dist_r = 0
    #         else:
    #             dist_r = 1-math.exp(-(((x-dist_offset)/dist_scale)**2))
    #         return dist_r
        
    #     delta_x, delta_y, _, _, _, _ = super().get_delta_values(env, agent_id, in_km=True)
    #     d = 0.5*delta_loc(delta_x) + 0.5*delta_loc(delta_y)

    #     # def boundary_pen(d):
    #     #     # d in [KM]. Only contributes when margin to edge < bnd_offset.
    #     #     margin = self.half_map - d
    #     #     bnd_offset = 5
    #     #     bnd_scale = 3
    #     #     if margin > bnd_offset:
    #     #         return 0
    #     #     return -(1 - math.exp(-(((bnd_offset - margin)/bnd_scale)**2)))
        
    #     # dist_lon = env.agents[agent_id].get_distance_to_origin_lon() / 1000
    #     # dist_lat = env.agents[agent_id].get_distance_to_origin_lat() / 1000
    #     # b = 0.5*boundary_pen(dist_lat) + 0.5*boundary_pen(dist_lon)

    #     # def altitude_pen(alt):
    #     #     # alt in [KM]. Only contributes when within alt_offset of low/high limits.
    #     #     alt_low = getattr(self.config, 'altitude_limit', 2500) / 1000
    #     #     alt_high = getattr(self.config, 'altitude_limit_max', 10000) / 1000
    #     #     alt_offset = 1.5
    #     #     alt_scale = 1.5
    #     #     low_margin = alt - alt_low
    #     #     high_margin = alt_high - alt
    #     #     low_pen = 0 if low_margin > alt_offset else -(1 - math.exp(-(((alt_offset - low_margin)/alt_scale)**2)))
    #     #     high_pen = 0 if high_margin > alt_offset else -(1 - math.exp(-(((alt_offset - high_margin)/alt_scale)**2)))
    #     #     return low_pen + high_pen
        
    #     # dist_alt = env.agents[agent_id].get_altitude() / 1000
    #     # a = altitude_pen(dist_alt)

    #     # def speed_rew(v):
    #     #     # v in [km/h]. Saturates as speed grows.
    #     #     v_scale = 1000
    #     #     v_offset = 500
    #     #     if v < v_offset:
    #     #         return 0
    #     #     return 1 - math.exp(-(((v - v_offset)/v_scale)**2))

    #     # velocity = env.agents[agent_id].get_norm_velocity_2d() * 3.6 # in km/h
    #     # v = speed_rew(velocity)

    #     # _, a_t, _, _, _, _ = super().get_orientation_metrics(env, agent_id, deg=True)
    #     # head_scale = 50
    #     # head_t = math.exp(-((a_t/head_scale)**2))

    #     # reward = d + b + 0.25*a + 0.25*v + 0.5*head_t

        
    #     # reward = d + b
    #     reward = d 
    #     return reward