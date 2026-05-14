import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from env.catalog import Catalog as c
from env.utils import get_AO_TA_R, get2d_AO_TA_R, LLA2NEU


class BaseRewardFunction(ABC):
    """
    Base RewardFunction class
    """
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger(__name__)
        self.reward_scale = getattr(self.config, f'{self.__class__.__name__}_scale', 1.0)
        self.is_potential = getattr(self.config, f'{self.__class__.__name__}_potential', False)
        self.pre_rewards = defaultdict(float)

    def reset(self, task, env):
        """
        Reward function-specific reset.
        """
        if self.is_potential:
            self.pre_rewards.clear()
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_learning:
                    self.pre_rewards[agent_id] = self.get_reward(task, env, agent_id)

    @abstractmethod
    def get_reward(self, task, env, agent_id, action=None):
        """
        Compute the reward at the current timestep.
        """
        raise NotImplementedError

    def _process(self, new_reward, agent_id, render_items=()):
        """Process reward and inner variables.
        """
        reward = new_reward * self.reward_scale
        if self.is_potential:
            reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward
        return reward
    
    def get_orientation_metrics(self, env, agent_id, deg=False, two_d=False):
        """
        Get posture to closest enemy aircraft.
        """
        state_vars = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.velocities_v_north_mps,           # 3. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 4. v_east     (unit: m/s)
            c.velocities_v_up_mps,              # 5. v_up     (unit: m/s)
            c.attitude_roll_rad,                 # 6. roll     (deg)
            c.velocities_p_rad_sec,              # 7. roll rate
            c.velocities_q_rad_sec              # 8. pitch rate
        ]
        ego_state = np.array(env.agents[agent_id].get_property_values(state_vars))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[3:6])])

        opp_state = np.array(env.agents[agent_id].opponents[0].get_property_values(state_vars))
        opp_cur_ned = LLA2NEU(*opp_state[:3], env.center_lon, env.center_lat, env.center_alt)
        opp_feature = np.array([*opp_cur_ned, *(opp_state[3:6])])
        if two_d:
            a_a, a_t, dist = get2d_AO_TA_R(ego_feature, opp_feature, return_side=False)
        else:
            a_a, a_t, dist = get_AO_TA_R(ego_feature, opp_feature, return_side=False)
        if deg:
            a_a, a_t, a_r, a_rr, a_pr = np.rad2deg(a_a), np.rad2deg(a_t), np.rad2deg(ego_state[6]), np.rad2deg(ego_state[7]), np.rad2deg(ego_state[8])
        else:
            a_r, a_rr, a_pr = ego_state[6], ego_state[7], ego_state[8]
        return a_a, a_t, a_r, a_rr, a_pr, dist

    def get_delta_values(self, env, agent_id, in_km=True):
        """
        delta_xyz -> in [m] -> if in_km: in [km]
        deltat_vx_vy_vz -> in [m/s]
        """
        state_vars = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.velocities_v_north_mps,           # 3. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 4. v_east     (unit: m/s)
            c.velocities_v_up_mps,              # 5. v_up     (unit: m/s)
        ]
        ego_state = np.array(env.agents[agent_id].get_property_values(state_vars))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[3:6])])

        opp_state = np.array(env.agents[agent_id].opponents[0].get_property_values(state_vars))
        opp_cur_ned = LLA2NEU(*opp_state[:3], env.center_lon, env.center_lat, env.center_alt)
        opp_feature = np.array([*opp_cur_ned, *(opp_state[3:6])])

        ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
        enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = opp_feature
        delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
        delta_vx, delta_vy, delta_vz = enm_vx - ego_vx, enm_vy - ego_vy, enm_vz - ego_vz
        if in_km:
            return abs(delta_x/1000), abs(delta_y/1000), abs(delta_z/1000), abs(delta_vx), abs(delta_vy), abs(delta_vz)
        else:
            return abs(delta_x), abs(delta_y), abs(delta_z), abs(delta_vx), abs(delta_vy), abs(delta_vz)