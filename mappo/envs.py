import numpy as np

class AircombatWrapper:
    def __init__(
        self,
        env_name=None,
        reward_aggr="mean",
        shared_rewards=False,
        agent_ids=False,
        global_state=True,
        seed=None,
        **kwargs,
    ):
        from env.env_base import AircraftEnv
        # from env.env_base_Backup import AircraftEnv

        config = {"mission_config": env_name, **kwargs} # kwargs contains hierarchy_level and training_level

        self.env = AircraftEnv(config)
        if seed is not None:
            self.env.seed(seed)
        self.agent_keys = self.env.rl_agent_ids
        self.first_agent = self.agent_keys[0]
        self.n_agents = len(self.agent_keys)

        self.reward_aggr = reward_aggr
        self.shared_rewards = shared_rewards
        self.agent_ids = agent_ids
        self.use_global_state = global_state

    def reset(self, seed=None, options=None):
        return self.env.reset(seed, options)

    def step(self, actions):
        return self.env.step(actions)
    
    def get_obs_size(self):
        return self.env.observation_space[self.first_agent].shape[0]

    def get_state_size(self):
        return self.env.observation_space[self.first_agent].shape[0]

    def get_state(self):
        return self.env.get_obs(self.first_agent)[self.first_agent]

    def get_action_size(self):
        return list(self.env.action_space[self.first_agent].nvec)

    def get_action_space(self):
        return ("multi_discrete", list(self.env.action_space[self.first_agent].nvec))
    
    def get_avail_actions(self):
        return {aid: np.ones(len(self.env.action_space[aid].nvec), dtype=np.float32)
                for aid in self.agent_keys}

    def get_agents(self):
        return self.agent_keys

    def close(self):
        self.env.close()
    
def get_environment(env_type, env_name, agent_ids, kwargs):
    return AircombatWrapper(env_name=env_name, agent_ids=agent_ids, **kwargs)
