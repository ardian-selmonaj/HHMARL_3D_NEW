import numpy as np
import random
import os
import torch
from missions.mission_base import BaseMission
from ray.rllib.core.rl_module import RLModule

"""
Model for high-level commander training, that uses pre-trained L4 policies with mixed-strategy.
Compared to model_strategy.py, this class uses all pre-trained low-level control policies. 
This model represents the 'commander' of the opponents that follows a mixed-strategy behavior,
therefore it is very similar to task_hier.py
"""

class HierModel(BaseMission):
    def __init__(self, config):
        super().__init__(config) 
        self.opp_hier_strategy = getattr(config, 'opp_hier_strategy', "mixed")

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

    def get_obs(self, env, agent, **kwargs):
        """
        Non-RL agents do not need obs.
        """
        pass

    def set_policy(self, action, agent, env):
        """
        Set high-level action by selecting the low-level policy for the agent.
        """
        if self.opp_hier_strategy == "mixed":
            act = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2], k=1)[0]
        elif self.opp_hier_strategy == "aggressive":
            act = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1], k=1)[0]
        else:
            act = np.random.randint(0, 3)
        
        policy_type = self.hl_action_mapping[act]
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