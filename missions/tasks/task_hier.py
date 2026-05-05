import numpy as np
import os
import torch
from gymnasium import spaces
from missions.mission_base import BaseMission
from rewards import EventDrivenReward, ShootReward
from env.utils import get_AO_TA_R, LLA2NEU
from ray.rllib.core.rl_module import RLModule

"""
Task to learn high-level commander policy for strategic coordination. 
"""

class HierTask(BaseMission):
    def __init__(self, config):
        super().__init__(config)

    def load_model(self):
        """
        Load low-level policies.
        """
        self.policies = {'f16': {}, 'A4':{}}
        #pol_dirs = ['L4_cannon_f16', 'L4_cannon_A4', 'L4_track_f16', 'L4_track_A4', 'L2_escape_f16', 'L2_escape_A4']
        pol_dirs = ['L4_cannon_f16', 'L4_track_f16', 'L2_escape_f16']
        pol_name = 'default'
        for policy_name in pol_dirs:
            check = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', policy_name, 'learner_group', 'learner', 'rl_module', pol_name)
            policy = RLModule.from_checkpoint(check)
            action_dist_class = policy.get_inference_action_dist_cls()
            policy_type = policy_name.split("_")[1]
            if 'A4' in policy_name:
                self.policies['A4'][policy_type] = {'policy': policy, 'act_dist': action_dist_class}
            else:
                self.policies['f16'][policy_type] = {'policy': policy, 'act_dist': action_dist_class}

    def reset(self, env):
        assert self.num_rl_agents + self.num_opp_agents == len(env.rl_agent_ids) + len(env.opp_agent_ids), f"Specified num agents {self.num_rl_agents + self.num_opp_agents} does not match env RL ids + Opp ids {len(env.rl_agent_ids) + len(env.opp_agent_ids)}. Adjust your configuration!"
        self.policy_to_use = {aid: None for aid in env.agents.keys()}
        return super().reset(env)
    
    def step(self, env, agent):
        if agent.cannon.done: agent.cannon.revert_cannon()

    def load_rewards(self):
        self.rewards = [
            EventDrivenReward(self.config),
            ShootReward(self.config)
        ]

    def load_act_space(self):
        self.action_space = spaces.Discrete(3)
        self.hl_action_mapping = {0: 'cannon', 1: 'track', 2: 'escape'}
    
    def load_obs_space(self):
        self.num_rl_agents = getattr(self.config, 'num_rl_agents', 2)
        self.num_opp_agents = getattr(self.config, 'num_opp_agents', 2)
        self.opp_obs = self.fri_obs = 6
        self.opp_obs_len = self.opp_obs * (self.num_opp_agents)
        self.fri_obs_len = self.fri_obs * (self.num_rl_agents-1)
        self.own_obs_len = 8
        # self.observation_space = spaces.Dict({
        #     "own": spaces.Box(low=-1, high=1., shape=(self.own_obs_len,)),
        #     "fri": spaces.Box(low=-1, high=1., shape=(self.fri_obs_len,)),
        #     "opp": spaces.Box(low=-1, high=1., shape=(self.opp_obs_len,)),
        #     }
        # )
        self.observation_space = spaces.Box(low=-1, high=1., shape=(self.own_obs_len + self.opp_obs_len + self.fri_obs_len,))

    def get_obs(self, env, agent, **kwargs):
        """
        Convert simulation states into the format of observation_space
        ------
            all obs between [-1, 1]

            EGO: 8
            0. north            (10km) -> /25000 (map_size /2)
            1. east             (10km)
            2. altitude         (km) -> /10000
            3. heading (yaw)    (norm deg) -> /360
            4. ego v_north      (norm m/s) -> /700
            5. ego v_east       (norm m/s) -> /700
            6. ego v_up        (norm m/s) -> /700
            7. ego_vc          (norm m/s) -> /1400

            OPP & Fri: 6
            0. north            (10km) -> /25000 (map_size /2 *1000)
            1. east             (10km) -> /25000 (map_size /2 *1000)
            2. rel. distance    (10km) -> /max diag dist
            3. AO               (norm deg) -> np.rad2deg() / 180
            4. TA               (norm deg) -> np.rad2deg() / 180
            5. side             -1 or 1
        """

        own_obs = np.zeros(self.own_obs_len, dtype=np.float32)
        fri_obs = np.zeros(self.fri_obs_len, dtype=np.float32)
        opp_obs = np.zeros(self.opp_obs_len, dtype=np.float32)

        ego_state = np.array(agent.get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])

        own_obs[0] = ego_cur_ned[0] / (self.map_size/2)
        own_obs[1] = ego_cur_ned[1] / (self.map_size/2)
        own_obs[2] = ego_cur_ned[2] / (self.altitude_max)
        own_obs[3] = np.rad2deg(ego_state[3])/360
        own_obs[4] = ego_state[6] / 700
        own_obs[5] = ego_state[7] / 700
        own_obs[6] = ego_state[8] / 700
        own_obs[7] = ego_state[9] / 1400

        opp_idx = 0
        opponents = super().get_closest_opp(agent)
        for opp, _ in opponents:
            if opp.alive:
                opp_state = np.array(opp.get_property_values(self.state_var))
                opp_cur_ned = LLA2NEU(*opp_state[:3], env.center_lon, env.center_lat, env.center_alt)
                opp_feature = np.array([*opp_cur_ned, *(opp_state[6:9])])
                a_a, a_t, dist, side = get_AO_TA_R(ego_feature, opp_feature, return_side=True)

                opp_obs[0+opp_idx*self.opp_obs] = opp_cur_ned[0] / (self.map_size/2)
                opp_obs[1+opp_idx*self.opp_obs] = opp_cur_ned[1] / (self.map_size/2)
                opp_obs[2+opp_idx*self.opp_obs] = dist / np.sqrt(self.map_size**2 + self.map_size**2 + self.altitude_delta**2)
                opp_obs[3+opp_idx*self.opp_obs] = np.rad2deg(a_a) / 180
                opp_obs[4+opp_idx*self.opp_obs] = np.rad2deg(a_t) / 180
                opp_obs[5+opp_idx*self.opp_obs] = side
                opp_idx += 1

        fri_idx = 0
        friendlies = super().get_closest_friendly(agent)
        for fri, _ in friendlies:
            if fri.alive:
                fri_state = np.array(fri.get_property_values(self.state_var))
                fri_cur_ned = LLA2NEU(*fri_state[:3], env.center_lon, env.center_lat, env.center_alt)
                fri_feature = np.array([*fri_cur_ned, *(fri_state[6:9])])
                a_a, a_t, dist, side = get_AO_TA_R(ego_feature, fri_feature, return_side=True)

                fri_obs[0+fri_idx*self.fri_obs] = fri_cur_ned[0] / (self.map_size/2)
                fri_obs[1+fri_idx*self.fri_obs] = fri_cur_ned[1] / (self.map_size/2)
                fri_obs[2+fri_idx*self.fri_obs] = dist / np.sqrt(self.map_size**2 + self.map_size**2 + self.altitude_delta**2)
                fri_obs[3+fri_idx*self.fri_obs] = np.rad2deg(a_a) / 180
                fri_obs[4+fri_idx*self.fri_obs] = np.rad2deg(a_t) / 180
                fri_obs[5+fri_idx*self.fri_obs] = side
                fri_idx += 1
        
        assert not np.isnan(own_obs).any(), self.log.error(f"Own Observation contains NaN values: {own_obs}")
        assert not np.isnan(fri_obs).any(), self.log.error(f"Fri Observation contains NaN values: {fri_obs}")
        assert not np.isnan(opp_obs).any(), self.log.error(f"Opp Observation contains NaN values: {opp_obs}")

        #return {"own": own_obs, "fri": fri_obs, "opp": opp_obs}
        return np.concatenate((own_obs, opp_obs, fri_obs), axis=0, dtype=np.float32)

    def set_policy(self, action, agent, env):
        """
        Set high-level action by selecting the low-level policy for the agent.
        """
        policy_type = self.hl_action_mapping[action]
        self.policy_to_use[agent.aid] = self.policies[agent.model][policy_type]
   
    def run_action(self, agent, env):
        """
        Execute low-level actions.
        """

        ll_obs = super().get_obs(env=env, agent=agent)
        ll_action = self.get_ll_action(ll_obs, agent)
        if bool(ll_action[4]): self._cannon(agent, env)
        ll_action = super().get_action(action=ll_action, agent=agent, env=env)
        agent.set_property_values(self.action_var, ll_action)
        for _ in range(env.sim_steps):
            agent.run()

    def get_ll_action(self, obs, agent):
        """
        Compute low-level actions.
        """
        try:
            logits = self.policy_to_use[agent.aid]['policy'].forward_inference({"obs": torch.from_numpy(obs).unsqueeze(0)})
        except:
            logits = None
            raise ValueError(f"Could not compute logits, check input: {torch.from_numpy(obs).unsqueeze(0)}")
        action_distribution = self.policy_to_use[agent.aid]['act_dist'].from_logits(logits["action_dist_inputs"]).to_deterministic()
        action = action_distribution.sample().detach().cpu().flatten().numpy()
        return action
    
    def _cannon(self, agent, env):
        """
        Handle cannon action.
        """
        if agent.alive and agent.cannon.ready:
            agent.cannon.launch(parent=agent, env=env)