import logging
import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from env.catalog import Catalog as c
from terminations import ExtremeState, Overload, LowAltitude, Timeout, DeadAgent
from env.utils import get_AO_TA_R, get2d_AO_TA_R, LLA2NEU, get_separate_deltas

LAST_ACTIONS = False

class BaseMission(ABC):
    """
    Base Mission class. A Mission is either 
        1) A task for trainable RL agents
        2) A model, either pre-trained or rule-based, i.e. NOT trainable agents.
    BaseMission is pre-defined for low-level policy training.
    Hierarchy subclasses specifically define observations, actions, termination and rewards.
    """
    def __init__(self, config):
        self.log = logging.getLogger(__name__)
        self.config = config
        self.shift = 4 if LAST_ACTIONS else 0
        self.load()

    def load(self):
        """
        Load specific mission objects.
        """
        opp_obs_len = 13
        self.obs_len = 17 + self.shift + opp_obs_len
        
        self.velocity_max = 670
        self.map_size = getattr(self.config, 'map_size', 50) * 1000 # map size in m
        self.loc_scale, self.loc_offset = self.shift_range(-self.map_size/2, self.map_size/2, -1, 1)
        self.loc_delta_scale, self.loc_delta_offset = self.shift_range(-self.map_size, self.map_size, -1, 1)

        self.altitude_min = getattr(self.config, 'altitude_limit', 2500)
        self.altitude_max = getattr(self.config, 'altitude_limit_max', 10000)
        self.alt_scale, self.alt_offset = self.shift_range(self.altitude_min, self.altitude_max, 0, 1)
        self.altitude_delta = self.altitude_max - self.altitude_min
        self.alt_delta_scale, self.alt_delta_offset = self.shift_range(-self.altitude_delta, self.altitude_delta, -1, 1)

        self.vel_scale, self.vel_offset = self.shift_range(-self.velocity_max, self.velocity_max, -1, 1)
        self.vel_delta_scale, self.vel_delta_offset = self.shift_range(-2*self.velocity_max, 2*self.velocity_max, -1, 1)

        self.aspect_target_scale, self.aspect_target_offset = self.shift_range(0, np.pi, 0, 1)

        self.load_obs_space()
        self.load_obs_vars()
        self.load_act_space()
        self.load_act_vars()
        self.load_terminations()
        self.load_rewards()
        self.load_model()
        self.load_render_vars()

    @abstractmethod
    def load_rewards(self):
        """
        Load base reward functions.
        """
        raise NotImplementedError

    def load_obs_space(self):
        # self.observation_space = spaces.Box(low=-1, high=1., shape=(self.obs_len,))
        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.obs_len,))

    def load_obs_vars(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. longitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)

            c.velocities_v_north_mps,           # 3. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 4. v_east     (unit: m/s)
            c.velocities_v_up_mps,              # 5. v_up       (unit: m/s)

            c.velocities_vc_mps,                # 6. vc (airspeed)  (unit: m/s)
            c.aero_alpha_rad,                   # 7. angle of attack (unit: rad)
            c.aero_beta_rad,                    # 8. sideslip angle (unit: rad)

            c.velocities_p_rad_sec,             # 9. roll rate     (unit: rad/s)
            c.velocities_q_rad_sec,             # 10. pitch rate    (unit: rad/s)
            c.velocities_r_rad_sec,             # 11. yaw rate      (unit: rad/s)

            c.attitude_phi_rad,                 # 12. roll/phi       (unit: rad)
            c.attitude_theta_rad,               # 13. pitch/theta    (unit: rad)
            c.attitude_psi_rad,                 # 14. yaw/psi        (unit: rad)
        ]

    def get_obs(self, env, agent, **kwargs):
        """
        TRACK --> extended with missile and shot.
        17 own
        13 opp
        """

        norm_obs = np.zeros(self.obs_len, dtype=np.float32)
        ego_state = np.array(agent.get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[3:6])])

        norm_obs[0] = ego_cur_ned[0] * self.loc_scale + self.loc_offset         #north
        norm_obs[1] = ego_cur_ned[1] * self.loc_scale + self.loc_offset         #east
        norm_obs[2] = ego_cur_ned[2] * self.alt_scale + self.alt_offset         #altitude
        norm_obs[3] = ego_state[3] * self.vel_scale + self.vel_offset           #north velocities
        norm_obs[4] = ego_state[4] * self.vel_scale + self.vel_offset           #east velocities
        norm_obs[5] = ego_state[5] * self.vel_scale + self.vel_offset           #up velocity
        norm_obs[6:9] = ego_state[9:12]                                         #angular rates of roll, pitch and yaw in [rad/sec]
        norm_obs[9], norm_obs[10] = np.cos(ego_state[12]), np.sin(ego_state[12])    #cos/sin of roll/phi
        norm_obs[11], norm_obs[12] = np.cos(ego_state[13]), np.sin(ego_state[13])   #cos/sin of pitch/theta
        norm_obs[13], norm_obs[14] = np.cos(ego_state[14]), np.sin(ego_state[14])   #cos/sin of yaw/psi

        norm_obs[15] = int(agent.cannon.alive)
        norm_obs[16] = agent.cannon.ammunition

        if LAST_ACTIONS:
            acts = np.array(agent.get_property_values(self.action_var))
            norm_obs[17:21] = acts

        opponent = self.get_closest_opp(agent)
        if opponent:
            opponent = opponent[0][0]
            opp_state = np.array(opponent.get_property_values(self.state_var))
            opp_cur_ned = LLA2NEU(*opp_state[:3], env.center_lon, env.center_lat, env.center_alt)
            opp_feature = np.array([*opp_cur_ned, *(opp_state[3:6])])
            a_a, a_t, _, side = get_AO_TA_R(ego_feature, opp_feature, return_side=True)
            delta_x, delta_y, delta_z, delta_vx, delta_vy, delta_vz = get_separate_deltas(ego_feature, opp_feature)

            norm_obs[17+self.shift] = delta_x * self.loc_delta_scale + self.loc_delta_offset
            norm_obs[18+self.shift] = delta_y * self.loc_delta_scale + self.loc_delta_offset
            norm_obs[19+self.shift] = delta_z * self.alt_delta_scale + self.alt_delta_offset

            norm_obs[20+self.shift] = delta_vx * self.vel_delta_scale + self.vel_delta_offset
            norm_obs[21+self.shift] = delta_vy * self.vel_delta_scale + self.vel_delta_offset
            norm_obs[22+self.shift] = delta_vz * self.vel_delta_scale + self.vel_delta_offset

            norm_obs[23+self.shift] = a_a * self.aspect_target_scale + self.aspect_target_offset
            norm_obs[24+self.shift] = a_t * self.aspect_target_scale + self.aspect_target_offset
            norm_obs[25+self.shift] = side

            norm_obs[26+self.shift], norm_obs[27+self.shift] = np.cos(opp_state[13]), np.sin(opp_state[13])  #cos/sin of pitch/theta
            norm_obs[28+self.shift], norm_obs[29+self.shift] = np.cos(opp_state[14]), np.sin(opp_state[14])  #cos/sin of yaw/psi

        assert not np.isnan(norm_obs).any(), self.log.error(f"Observation contains NaN values: {norm_obs}")

        return norm_obs

    def load_act_space(self):
        """
        Load action space.
        """
        # aileron, elevator, rudder, throttle, shoot (bool)
        # self.action_space = spaces.MultiDiscrete([31, 31, 31, 31, 2])
        self.action_space = spaces.MultiDiscrete([21, 11, 9, 9, 2])
    
    def load_act_vars(self):
        """
        Load action parameters.
        """
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.], roll
            c.fcs_elevator_cmd_norm,            # [-1., 1.], pitch
            c.fcs_rudder_cmd_norm,              # [-1., 1.], yaw
            c.fcs_throttle_cmd_norm,            # [0.3, 0.9], power, speed
        ]

    def load_terminations(self):
        """
        Load base termination conditions.
        """
        self.terminations = [
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            #Timeout(self.config),
            DeadAgent(self.config)
        ]

    def load_render_vars(self):
        """
        Load parameters for rendering.
        """
        self.render_var = [
            c.position_lat_gc_rad,
            c.position_long_gc_rad,
            c.position_h_sl_m,
            c.attitude_phi_rad,
            c.attitude_theta_rad,
            c.attitude_psi_rad,
        ]

    def load_model(self):
        """
        Load pre-trained or rule-based model for non-rl agents.
        """
        pass

    def reset(self, env):
        """
        Task-specific reset
        """
        for rew in self.rewards:
            rew.reset(self, env)

    def step(self, env, agent):
        """ 
        Task-specific step
        """
        pass

    def get_action(self, action, agent, env):
        """
        Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)

        ### ABS ACT
        act_space = spaces.MultiDiscrete([21, 11, 9, 9, 2])
        scale, offset = self.shift_range(0, act_space.nvec[0]-1, -0.75, 0.75)
        norm_act[0] = action[0]*scale + offset
        scale, offset = self.shift_range(0, act_space.nvec[1]-1, -0.75, 0.75)
        norm_act[1] = action[1]*scale + offset
        scale, offset = self.shift_range(0, act_space.nvec[2]-1, -0.5, 0.5)
        norm_act[2] = action[2]*scale + offset
        scale, offset = self.shift_range(0, act_space.nvec[3]-1, 0.3, 0.9)
        norm_act[3] = action[3]*scale + offset


        ### REL ACT
        # act_space = spaces.MultiDiscrete([11, 11, 11, 11, 2])
        # scale, offset = self.shift_range(0, act_space.nvec[0]-1, -0.5, 0.5)
        # norm_act[0] = action[0]*scale + offset
        # norm_act[1] = action[1]*scale + offset
        # norm_act[2] = action[2]*scale + offset
        # norm_act[3] = action[3]*scale + offset

        return norm_act

    def get_reward(self, env, agent_id, action):
        """
        Aggregate reward functions
        """
        reward = 0.0
        ##### ---> 'if alive' prevents dead agents from getting a reward/punish.
        # if env.agents[agent_id].alive:
        for rew in self.rewards:
            reward += rew.get_reward(self, env, agent_id, action)
        return np.round(reward, 5)

    def get_termination(self, env, agent_id, info={}, get_info=False):
        """
        Aggregate termination conditions
        """
        done = not env.agents[agent_id].alive
        success = True
        if not done:
            for condition in self.terminations:
                d, s, info = condition.get_termination(self, env, agent_id, info)
                done = done or d
                success = success and s
                if done:
                    break

        if get_info:
            return done, info
        else:
            return done

    def get_closest_opp(self, agent):
        """
        Sort alive opponents by distance.
        """
        opps = [
            [opp, agent.get_distance_to_target(opp)]
            for opp in agent.opponents if opp.alive
        ]
        return sorted(opps, key=lambda x: x[1])
        
    def get_closest_friendly(self, agent):
        """
        Sort friendly aircraft by distance.
        """
        fris = []
        for fri in agent.partners:
            fris.append([fri, agent.get_distance_to_target(fri)])
        sorted_fris = sorted(fris, key=lambda x: x[1])
        return sorted_fris
    
    def shift_range(self, a, b, c, d):
        """
        Compute scale and offset for mapping a value from range [a, b] to range [c, d].

        The transformation will be:
            mapped_value = original_value * scale + offset
        """
        scale = (d - c) / (b - a)
        offset = c - a * scale
        return scale, offset
