import torch
import os
import numpy as np
import random
from gymnasium import spaces
from env.catalog import Catalog as c
from missions.mission_base import BaseMission
from missions.models.actors.baseline_actor import BaselineActor
from env.utils import get_root_dir, in_range_rad

"""
Pre-trained agent to track (pursue) given target.
"""

class TrackModel(BaseMission):
    def __init__(self, config) -> None:
        super().__init__(config)

    def load_act_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def load_obs_space(self):
        """
        Non-RL agents do not need obs spaces.
        """
        pass

    def load_obs_vars(self):
        self.state_var = [
            c.delta_altitude,                   #  0. delta_h   (unit: m)
            c.delta_heading,                    #  1. delta_heading  (unit: °)
            c.delta_velocities_u,               #  2. delta_v   (unit: m/s)
            c.attitude_roll_rad,                #  3. roll      (unit: rad)
            c.attitude_pitch_rad,               #  4. pitch     (unit: rad)
            c.velocities_u_mps,                 #  5. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 #  6. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 #  7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                #  8. vc        (unit: m/s)
            c.position_h_sl_m                   #  9. altitude  (unit: m)
        ]

    def load_rewards(self):
        """
        No rewards needed for non-rl agents.
        """
        self.rewards = []

    def load_model(self):
        """
        Load pre-trained model.
        """
        self.model_path = os.path.join(get_root_dir(), "missions", "models", "actors", "baseline_actor.pt")
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        self.actor.eval()

    def reset(self, env):
        """
        Model specific reset.
        """
        self.rnn_states = np.zeros((1, 1, 128))
        super().reset(env)

    def get_obs(self, env:object, agent: object, **kwargs):
        delta_value = kwargs["kwargs"].get("delta", None)
        assert delta_value is not None, self.log.error(f"Delta Value in {self.__class__.__name__} is missing.")
        obs = agent.get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000          #  0. ego delta altitude  (unit: 1km)
        norm_obs[1] = in_range_rad(delta_value[1])   #  1. ego delta heading   (unit rad)
        norm_obs[2] = delta_value[2] / 340           #  2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = obs[9] / 5000                  #  3. ego_altitude (unit: km)
        norm_obs[4] = np.sin(obs[3])                 #  4. ego_roll_sin
        norm_obs[5] = np.cos(obs[3])                 #  5. ego_roll_cos
        norm_obs[6] = np.sin(obs[4])                 #  6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[4])                 #  7. ego_pitch_cos
        norm_obs[8] = obs[5] / 340                   #  8. ego_v_x   (unit: mh)
        norm_obs[9] = obs[6] / 340                   #  9. ego_v_y    (unit: mh)
        norm_obs[10] = obs[7] / 340                  #  10. ego_v_z    (unit: mh)
        norm_obs[11] = obs[8] / 340                  #  11. ego_vc        (unit: mh)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs
    
    def get_action(self, action=None, agent=None, env=None):
        delta_value = self.set_delta_value(agent)
        observation = self.get_obs(None, agent, kwargs={"delta": delta_value})
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return self.shift_action(action)
    
    def shift_action(self, action):
        """
        Model was trained under different action space and ranges.
        """
        norm_act = np.zeros(4)

        scale, offset = self.shift_range(0, self.action_space.nvec[0]-1, -1, 1) # 0~40 => -1~1
        norm_act[0] = action[0]*scale + offset
        norm_act[1] = action[1]*scale + offset
        norm_act[2] = action[2]*scale + offset

        scale, offset = self.shift_range(0, self.action_space.nvec[3]-1, 0.4, 0.9) # 0~29 => 0.4~0.9
        norm_act[3] = action[3]*scale + offset

        return norm_act

    def set_delta_value(self, agent: object):
        ego_x, ego_y, ego_z = agent.get_position()
        ego_vx, ego_vy, ego_vz = agent.get_velocity()
        opp_x, opp_y, opp_z = agent.opponents[0].get_position()
        # delta altitude
        delta_altitude = opp_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = opp_x - ego_x, opp_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = agent.opponents[0].get_property_value(c.velocities_u_mps) - \
                         agent.get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])   


class L1maneuverModel(TrackModel):
    """
    Randomly turn to right and change velocity & heading.
    Circling agent.
    """
    def __init__(self, config,):
        super().__init__(config)

        self.turn_interval = 40
        self.step_interval =  self.turn_interval / 0.1 #time_interval

        self.target_heading_list = [np.pi/3, np.pi, -np.pi/3]
        self.target_altitude = 6000
        self.target_velocity = 250

    def reset(self, env):
        self._step = 0
        self._i = 0
        self._rnd_head = 0
        self._rnd_vel = 0
        self.init_heading = None
        super().reset(env)
    
    def set_delta_value(self, agent):
        cur_heading = agent.get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading

        if self._step % self.step_interval == 0:
            self._i = (self._i + 1) %3
            self._rnd_head = np.random.randint(-700, 700)
            self._rnd_vel = np.random.randint(-20, 20)

        delta_heading = self.init_heading + self.target_heading_list[self._i] - cur_heading
        delta_altitude = self.target_altitude +self._rnd_head - agent.get_property_value(c.position_h_sl_m)
        delta_velocity = self.target_velocity +self._rnd_vel - agent.get_property_value(c.velocities_u_mps)
        self._step += 1
        return np.array([delta_altitude, delta_heading, delta_velocity])


class L2maneuverModel(TrackModel):
    """
    Randomly turn to left or right and change velocity & heading.
    Randomly shoot with cannon.
    """
    def __init__(self, config,):
        super().__init__(config)

        self.turn_interval = np.random.randint(20, 60)
        self.step_interval =  self.turn_interval / 0.1 #time_interval

        self.target_heading_list = [np.pi/3, np.pi, -np.pi/3]
        self.target_altitude = 6000
        self.target_velocity = 250

    def reset(self, env):
        self._step = 0
        self._i = 0
        self._rnd_head = 0
        self._rnd_vel = 0
        self.turn_interval = np.random.randint(20, 60)
        self.step_interval =  self.turn_interval / 0.1 #time_interval
        self.init_heading = np.random.choice([0, 2*np.pi])
        super().reset(env)
    
    def set_delta_value(self, agent):
        cur_heading = agent.get_property_value(c.attitude_heading_true_rad)

        if self._step % self.step_interval == 0:
            self._i = (self._i + 1) %3 #np.random.randint(0,3)
            self._rnd_head = np.random.randint(-900, 900)
            self._rnd_vel = np.random.randint(-30, 30)

        if self.init_heading:
            delta_heading = self.init_heading - self.target_heading_list[self._i] - cur_heading
        else:
            delta_heading = self.target_heading_list[self._i] - cur_heading
        delta_altitude = self.target_altitude +self._rnd_head - agent.get_property_value(c.position_h_sl_m)
        delta_velocity = self.target_velocity +self._rnd_vel - agent.get_property_value(c.velocities_u_mps)
        self._step += 1
        return np.array([delta_altitude, delta_heading, delta_velocity])

    def get_action(self, action=None, agent=None, env=None):
        if agent.cannon.ready and self._step % self.step_interval in range(100,130) and random.choices([True, False], weights=[0.4, 0.6], k=1)[0]:
            agent.cannon.launch(parent=agent, env=env)
        return super().get_action(action, agent, env)

    def step(self, env, agent):
        """
        Cannon reset if hit.
        """
        if agent.cannon.done: agent.cannon.revert_cannon()


class L3maneuverModel(TrackModel):
    """
    An agent for tracking and shooting (precisely) with cannon.
    """

    def __init__(self, config,):
        super().__init__(config)

    def reset(self, env):
        self.rnd_alt = random.uniform(-300, 300)
        self.rnd_head = random.uniform(-30, 30)
        self.rnd_vel = random.uniform(-100, 100)
        super().reset(env)
    
    def get_action(self, action=None, agent=None, env=None):
        """
        Slightly perturb direction vector (delta_values) to induce some randomness.
        delta[0] = altitude in m
        delta[1] = heading in rad
        delta[3] = velocity in m/s
        """
        if agent.cannon.ready:
            self.check_cannon(agent, env)

        delta_value = self.set_delta_value(agent)
        delta_value[0] = delta_value[0] + self.rnd_alt
        delta_value[1] = delta_value[1] + np.deg2rad(self.rnd_head)
        delta_value[2] = delta_value[2] + self.rnd_vel

        observation = self.get_obs(None, agent, kwargs={"delta": delta_value})
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return self.shift_action(action)
    
    def check_cannon(self, agent, env):
        """
        Check if opponent is close to be within Weapon Engagement Zone of cannon,
        wich enlarged omega 5°, and enlarged distance +500m.
        """
        ata, dist = agent.cannon.get_distance_angle(agent=agent, target=agent.opponents[0])
        if (abs(ata) < agent.cannon.omega_wez+5) and (dist < agent.cannon.Rmax_wez+500):
            agent.cannon.launch(parent=agent, env=env)
    
    def step(self, env, agent):
        """
        Cannon reset if hit.
        """
        if agent.cannon.done: agent.cannon.revert_cannon()