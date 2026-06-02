import logging
import os
import numpy as np
import torch
from abc import ABC, abstractmethod
from gymnasium import spaces

from env.catalog import Catalog as c
from env.utils import get_AO_TA_R, LLA2NEU, get_separate_deltas, load_pretrained_actors
from env.terminations import ExtremeState, Overload, LowAltitude, DeadAgent
from env.rewards import (
    EventDrivenReward,
    ShootReward,
    ShootPenaltyReward,
    HeadingReward,
    DistanceReward,
    FleeReward,
)


LAST_ACTIONS = False


class BaseMission(ABC):
    """
    A Mission is either a task for trainable RL agents or a model
    (pre-trained or rule-based) for non-trainable agents.
    BaseMission defines the low-level observation/action/termination/reward layout.
    """

    def __init__(self, config):
        self.log = logging.getLogger(__name__)
        self.config = config
        self.shift = 4 if LAST_ACTIONS else 0
        self.load()

    def load(self):
        opp_obs_len = 13
        self.obs_len = 17 + self.shift + opp_obs_len

        self.velocity_max = 670
        self.map_size = getattr(self.config, 'map_size', 50) * 1000
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
        raise NotImplementedError

    def load_obs_space(self):
        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.obs_len,))

    def load_obs_vars(self):
        self.state_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.velocities_v_north_mps,
            c.velocities_v_east_mps,
            c.velocities_v_up_mps,
            c.velocities_vc_mps,
            c.aero_alpha_rad,
            c.aero_beta_rad,
            c.velocities_p_rad_sec,
            c.velocities_q_rad_sec,
            c.velocities_r_rad_sec,
            c.attitude_phi_rad,
            c.attitude_theta_rad,
            c.attitude_psi_rad,
        ]

    def get_obs(self, env, agent, **kwargs):
        norm_obs = np.zeros(self.obs_len, dtype=np.float32)
        ego_state = np.array(agent.get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[3:6])])

        norm_obs[0] = ego_cur_ned[0] * self.loc_scale + self.loc_offset
        norm_obs[1] = ego_cur_ned[1] * self.loc_scale + self.loc_offset
        norm_obs[2] = ego_cur_ned[2] * self.alt_scale + self.alt_offset
        norm_obs[3] = ego_state[3] * self.vel_scale + self.vel_offset
        norm_obs[4] = ego_state[4] * self.vel_scale + self.vel_offset
        norm_obs[5] = ego_state[5] * self.vel_scale + self.vel_offset
        norm_obs[6:9] = ego_state[9:12]
        norm_obs[9], norm_obs[10] = np.cos(ego_state[12]), np.sin(ego_state[12])
        norm_obs[11], norm_obs[12] = np.cos(ego_state[13]), np.sin(ego_state[13])
        norm_obs[13], norm_obs[14] = np.cos(ego_state[14]), np.sin(ego_state[14])

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

            norm_obs[26+self.shift], norm_obs[27+self.shift] = np.cos(opp_state[13]), np.sin(opp_state[13])
            norm_obs[28+self.shift], norm_obs[29+self.shift] = np.cos(opp_state[14]), np.sin(opp_state[14])

        assert not np.isnan(norm_obs).any(), self.log.error(f"Observation contains NaN values: {norm_obs}")
        return norm_obs

    def load_act_space(self):
        # aileron, elevator, rudder, throttle, shoot (bool)
        self.action_space = spaces.MultiDiscrete([17, 11, 9, 9, 2])

    def load_act_vars(self):
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]

    def load_terminations(self):
        self.terminations = [
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            DeadAgent(self.config),
        ]

    def load_render_vars(self):
        self.render_var = [
            c.position_lat_gc_rad,
            c.position_long_gc_rad,
            c.position_h_sl_m,
            c.attitude_phi_rad,
            c.attitude_theta_rad,
            c.attitude_psi_rad,
        ]

    def load_model(self):
        """Load pre-trained or rule-based model for non-rl agents."""
        pass

    def reset(self, env):
        for rew in self.rewards:
            rew.reset(self, env)

    def step(self, env, agent):
        pass

    def get_action(self, action, agent, env):
        """Convert discrete action index into continuous value."""
        norm_act = np.zeros(4)
        act_space = spaces.MultiDiscrete([17, 11, 9, 9, 2])
        scale, offset = self.shift_range(0, act_space.nvec[0]-1, -0.8, 0.8)
        norm_act[0] = action[0]*scale + offset
        scale, offset = self.shift_range(0, act_space.nvec[1]-1, -0.8, 0.8)
        norm_act[1] = action[1]*scale + offset
        scale, offset = self.shift_range(0, act_space.nvec[2]-1, -0.5, 0.5)
        norm_act[2] = action[2]*scale + offset
        scale, offset = self.shift_range(0, act_space.nvec[3]-1, 0.3, 0.9)
        norm_act[3] = action[3]*scale + offset
        return norm_act

    def get_reward(self, env, agent_id, action):
        reward = 0.0
        for rew in self.rewards:
            reward += rew.get_reward(self, env, agent_id, action)
        return np.round(reward, 5)

    def get_termination(self, env, agent_id, info={}, get_info=False):
        done = not env.agents[agent_id].alive
        success = True
        if not done:
            for condition in self.terminations:
                d, s, info = condition.get_termination(self, env, agent_id, info)
                done = done or d
                success = success and s
                if done:
                    break
        return (done, info) if get_info else done

    def get_closest_opp(self, agent):
        opps = [[opp, agent.get_distance_to_target(opp)] for opp in agent.opponents if opp.alive]
        return sorted(opps, key=lambda x: x[1])

    def get_closest_friendly(self, agent):
        fris = [[fri, agent.get_distance_to_target(fri)] for fri in agent.partners]
        return sorted(fris, key=lambda x: x[1])

    def shift_range(self, a, b, c, d):
        scale = (d - c) / (b - a)
        offset = c - a * scale
        return scale, offset


class TrackTask(BaseMission):
    """Engage Task: track (pursue) target. Parent class of CannonTask and EscapeTask."""

    def load_rewards(self):
        self.rewards = [
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config),
            HeadingReward(self.config),
        ]

    def get_action(self, action, agent, env):
        shoot_action = action[4]
        action = super().get_action(action, agent, env)
        self._cannon(shoot_action, agent, env)
        return action

    def _cannon(self, shoot_action, agent, env):
        if agent.alive and shoot_action and agent.cannon.ready:
            self.shot_fired[agent.aid] = agent.cannon.launch(parent=agent, env=env)
        else:
            self.shot_fired[agent.aid] = False

    def reset(self, env):
        self.shot_fired = {aid: False for aid in env.agents.keys()}
        return super().reset(env)

    def step(self, env, agent):
        if agent.cannon.done:
            agent.cannon.revert_cannon()


class CannonTask(TrackTask):
    """Attack Task for aggressively destroying opponents."""

    def load_rewards(self):
        self.rewards = [
            ShootReward(self.config),
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config),
            HeadingReward(self.config),
        ]


class EscapeTask(TrackTask):
    """Defend Task to escape from opponent."""

    def load_rewards(self):
        self.rewards = [
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config),
            DistanceReward(self.config),
            HeadingReward(self.config),
            FleeReward(self.config),
        ]


class HierTask(BaseMission):
    """High-level commander policy that picks among pre-trained low-level policies."""

    def __init__(self, config):
        super().__init__(config)

    def load_rewards(self):
        self.rewards = [
            EventDrivenReward(self.config),
            ShootReward(self.config),
        ]

    def load_act_space(self):
        self.action_space = spaces.Discrete(3)
        self.hl_action_mapping = {0: 'cannon', 1: 'track', 2: 'escape'}

    def load_obs_space(self):
        self.num_rl_agents = getattr(self.config, 'num_rl_agents', 2)
        self.num_opp_agents = getattr(self.config, 'num_opp_agents', 2)
        self.opp_obs = self.fri_obs = 6
        self.opp_obs_len = self.opp_obs * self.num_opp_agents
        self.fri_obs_len = self.fri_obs * (self.num_rl_agents - 1)
        self.own_obs_len = 8
        self.observation_space = spaces.Box(
            low=-1, high=1.0,
            shape=(self.own_obs_len + self.opp_obs_len + self.fri_obs_len,),
        )

    def load_model(self):
        """Load low-level pre-trained policies keyed by aircraft model and task."""
        self.policies = self._build_low_level_policies()

    def _build_low_level_policies(self):
        """Build {model: {task: {'state_dict': ...}}} dict by loading checkpoints from runs/."""
        # pol_dirs = ['L4_cannon_f16', 'L4_cannon_A4', 'L4_track_f16', 'L4_track_A4', 'L2_escape_f16', 'L2_escape_A4']
        pol_dirs = ['L4_cannon_f16', 'L4_track_f16', 'L2_escape_f16']
        policies = {'f16': {}, 'A4': {}}
        for policy_name in pol_dirs:
            ckpt = load_pretrained_actors(policy_name)
            policy_type = policy_name.split('_')[1]
            model_type = 'A4' if 'A4' in policy_name else 'f16'
            policies[model_type][policy_type] = {'state_dict': ckpt['actors_state_dict']}
        return policies

    def reset(self, env):
        assert self.num_rl_agents + self.num_opp_agents == len(env.rl_agent_ids) + len(env.opp_agent_ids), \
            f"Specified num agents {self.num_rl_agents + self.num_opp_agents} does not match env RL ids + Opp ids {len(env.rl_agent_ids) + len(env.opp_agent_ids)}. Adjust your configuration!"
        self.policy_to_use = {aid: None for aid in env.agents.keys()}
        return super().reset(env)

    def step(self, env, agent):
        if agent.cannon.done:
            agent.cannon.revert_cannon()

    def get_obs(self, env, agent, **kwargs):
        own_obs = np.zeros(self.own_obs_len, dtype=np.float32)
        fri_obs = np.zeros(self.fri_obs_len, dtype=np.float32)
        opp_obs = np.zeros(self.opp_obs_len, dtype=np.float32)

        ego_state = np.array(agent.get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])

        own_obs[0] = ego_cur_ned[0] / (self.map_size/2)
        own_obs[1] = ego_cur_ned[1] / (self.map_size/2)
        own_obs[2] = ego_cur_ned[2] / self.altitude_max
        own_obs[3] = np.rad2deg(ego_state[3])/360
        own_obs[4] = ego_state[6] / 700
        own_obs[5] = ego_state[7] / 700
        own_obs[6] = ego_state[8] / 700
        own_obs[7] = ego_state[9] / 1400

        opp_idx = 0
        for opp, _ in super().get_closest_opp(agent):
            if not opp.alive:
                continue
            opp_state = np.array(opp.get_property_values(self.state_var))
            opp_cur_ned = LLA2NEU(*opp_state[:3], env.center_lon, env.center_lat, env.center_alt)
            opp_feature = np.array([*opp_cur_ned, *(opp_state[6:9])])
            a_a, a_t, dist, side = get_AO_TA_R(ego_feature, opp_feature, return_side=True)

            base = opp_idx * self.opp_obs
            opp_obs[base+0] = opp_cur_ned[0] / (self.map_size/2)
            opp_obs[base+1] = opp_cur_ned[1] / (self.map_size/2)
            opp_obs[base+2] = dist / np.sqrt(self.map_size**2 + self.map_size**2 + self.altitude_delta**2)
            opp_obs[base+3] = np.rad2deg(a_a) / 180
            opp_obs[base+4] = np.rad2deg(a_t) / 180
            opp_obs[base+5] = side
            opp_idx += 1

        fri_idx = 0
        for fri, _ in super().get_closest_friendly(agent):
            if not fri.alive:
                continue
            fri_state = np.array(fri.get_property_values(self.state_var))
            fri_cur_ned = LLA2NEU(*fri_state[:3], env.center_lon, env.center_lat, env.center_alt)
            fri_feature = np.array([*fri_cur_ned, *(fri_state[6:9])])
            a_a, a_t, dist, side = get_AO_TA_R(ego_feature, fri_feature, return_side=True)

            base = fri_idx * self.fri_obs
            fri_obs[base+0] = fri_cur_ned[0] / (self.map_size/2)
            fri_obs[base+1] = fri_cur_ned[1] / (self.map_size/2)
            fri_obs[base+2] = dist / np.sqrt(self.map_size**2 + self.map_size**2 + self.altitude_delta**2)
            fri_obs[base+3] = np.rad2deg(a_a) / 180
            fri_obs[base+4] = np.rad2deg(a_t) / 180
            fri_obs[base+5] = side
            fri_idx += 1

        assert not np.isnan(own_obs).any(), self.log.error(f"Own Observation contains NaN values: {own_obs}")
        assert not np.isnan(fri_obs).any(), self.log.error(f"Fri Observation contains NaN values: {fri_obs}")
        assert not np.isnan(opp_obs).any(), self.log.error(f"Opp Observation contains NaN values: {opp_obs}")
        return np.concatenate((own_obs, opp_obs, fri_obs), axis=0, dtype=np.float32)

    def set_policy(self, action, agent, env):
        policy_type = self.hl_action_mapping[action]
        self.policy_to_use[agent.aid] = self.policies[agent.model][policy_type]

    def run_action(self, agent, env):
        ll_obs = super().get_obs(env=env, agent=agent)
        ll_action = self.get_ll_action(ll_obs, agent)
        if bool(ll_action[4]):
            self._cannon(agent, env)
        ll_action = super().get_action(action=ll_action, agent=agent, env=env)
        agent.set_property_values(self.action_var, ll_action)
        for _ in range(env.sim_steps):
            agent.run()

    def get_ll_action(self, obs, agent):
        """Sample deterministic action from the chosen pre-trained low-level policy."""
        from algorithm import networks  # lazy: avoids env -> algorithm cycle at import time
        act_with_policies = networks.act_with_policies
        policy = self.policy_to_use[agent.aid]
        actors = policy.get('actors')
        if actors is None:
            raise RuntimeError("HierTask: pretrained low-level actors not instantiated. "
                               "Call attach_actor_modules() with env-specific shapes before running.")
        obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            actions, _, _, _ = act_with_policies(
                policy['args'], actors, obs_t, rnn_hxs=None, masks=None, deterministic=True,
            )
        return actions[0, 0].cpu().numpy()

    def _cannon(self, agent, env):
        if agent.alive and agent.cannon.ready:
            agent.cannon.launch(parent=agent, env=env)
