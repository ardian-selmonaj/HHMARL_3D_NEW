from terminations.termination_base import BaseTerminationCondition
import numpy as np
from env.catalog import Catalog as c
from env.utils import get_AO_TA_R, get2d_AO_TA_R, LLA2NEU

class HeadingReached(BaseTerminationCondition):
    """
    Heading reached
    """

    def __init__(self, config):
        super().__init__(config)
        

    def get_termination(self, task, env, agent_id, info={}):
        done = success = False
        a_a, a_t, a_r, dist = self.get_orientation_metrics(env, agent_id, deg=True)
        if a_a < 10 and a_t < 10 and dist < 5000:
            done = True
        return done, success, info


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
            c.attitude_roll_rad                 # 6. roll     (deg)
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
            a_a, a_t, a_r = np.rad2deg(a_a), np.rad2deg(a_t), np.rad2deg(ego_state[6])
        else:
            a_r = ego_state[6]
        return a_a, a_t, a_r, dist