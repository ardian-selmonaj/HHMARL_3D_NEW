import numpy as np
import random
import os
import torch
from missions.mission_base import BaseMission
from ray.rllib.core.rl_module import RLModule

"""
Model for L4 trainig, that uses pre-trained L3 policies with mixed-strategy.
"""

class StrategyModel(BaseMission):
    def __init__(self, config):
        super().__init__(config) 

    def load_model(self):
        """
        Load pre-trained policies.
        NOTE: in L4 training, we only use f16 model, as it performs better than A4
        """
        self.policies = {}
        pol_dirs = ['L3_cannon_f16', 'L3_track_f16', 'L1_escape_f16']
        pol_name = 'default'
        for policy_name in pol_dirs:
            check = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', policy_name, 'learner_group', 'learner', 'rl_module', pol_name)
            policy = RLModule.from_checkpoint(check)
            action_dist_class = policy.get_inference_action_dist_cls()
            policy_type = policy_name.split("_")[1]
            self.policies[policy_type] = {'policy': policy, 'act_dist': action_dist_class}

    def reset(self, env):
        self.policy_set_step = 0
        self.policy_to_use = {aid: None for aid in env.agents.keys()}
        return super().reset(env)
    
    def step(self, env, agent):
        if agent.cannon.done: agent.cannon.revert_cannon()

    def load_rewards(self):
        """
        No rewards needed for non-rl agents.
        """
        self.rewards = []

    def load_act_space(self):
        """
        Non-RL agents do not need obs spaces.
        """
        self.hl_action_mapping = {0: 'cannon', 1: 'track', 2: 'escape'}
    
    def load_obs_space(self):
        """
        Non-RL agents do not need obs spaces.
        """
        pass

    def get_action(self, action, agent, env):
        """
        Select control policy and execute action.
        """
        # change strategy with interval of (env.horizon / 5) steps.
        if self.policy_set_step == 0 or abs(env.steps - self.policy_set_step) > (env.horizon / 5):
            self.set_policy(action, agent, env)
            self.policy_set_step = env.steps

        assert self.policy_to_use[agent.aid], f"No policy was selected for agent {agent.aid}"

        obs = super().get_obs(env=env, agent=agent)
        act = self.get_policy_action(obs=obs, agent=agent)
        if bool(act[4]): self._cannon(agent, env)
        return super().get_action(action=act, agent=agent, env=env)

    def set_policy(self, action, agent, env):
        """
        Mixed Strategy for selecting the low-level policy for the agent.
        """
        act = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2], k=1)[0]
        policy_type = self.hl_action_mapping[act]
        self.policy_to_use[agent.aid] = self.policies[policy_type]
 
    def get_policy_action(self, obs, agent):
        """
        Compute action by chosen policy.
        """
        try:
            logits = self.policy_to_use[agent.aid]['policy'].forward_inference({"obs": torch.from_numpy(obs).unsqueeze(0)})
        except:
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