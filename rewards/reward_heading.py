import math
import numpy as np
from rewards.reward_base import BaseRewardFunction


class HeadingReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed']]
        self.with_opp = getattr(self.config, f'{self.__class__.__name__}_with_opp', True)
        self.roll_only = getattr(self.config, f'{self.__class__.__name__}_roll_only', False)

    def get_reward(self, task, env, agent_id, action=None):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable
        """
        reward = self.head_rew(env, agent_id)
        return self._process(reward, agent_id)

    def head_rew(self, env, agent_id):
        a_a, a_t, a_r, a_rr, a_pr, dist = super().get_orientation_metrics(env, agent_id, deg=True)
        delta_x, delta_y, delta_z, delta_vx, delta_vy, delta_vz = super().get_delta_values(env, agent_id, in_km=True)
        
        if self.with_opp:

            head_scale = 10
            head_a = math.exp(-((a_a/head_scale)**2))
            head_t = math.exp(-((a_t/(head_scale))**2))

            def delta_loc(x):
                # in [KM] 
                dist_scale = 4
                dist_offset = 2
                if x < dist_offset:
                    dist_r = 1
                else:
                    dist_r = math.exp(-(((x-dist_offset)/dist_scale)**4))
                return dist_r

            def delta_vel(x):
                # in [m/s]
                vel_scale = 100
                vel_offset = 30
                if x < vel_offset:
                    vel_r = 1
                else:
                    vel_r = math.exp(-(((x-vel_offset)/vel_scale)**2))
                return vel_r

            h = 0.35 * head_a + 0.65 * head_t
            d = 0.333 * delta_loc(delta_x) + 0.333 * delta_loc(delta_y) + 0.333 * delta_loc(delta_z)
            v = 0.5 * delta_vel(delta_vx) + 0.5 * delta_vel(delta_vy)

            # H0
            # roll_scale = 15
            # roll_r = math.exp(-((a_r/roll_scale)**2))
            # roll_rr = math.exp(-((a_rr/roll_scale)**2))
            # r = roll_r*roll_rr
            # reward = h * (d+v) + 1.0*head_a*head_t*r

            # # H1
            # r = ((roll_r-0.5)*2)*roll_rr
            # reward = h * (d+v) + head_a*head_t*r

            # H2
            # reward = h * (d+v) + 0.6*head_a*head_t*roll_rr + 0.4*((roll_r-0.5)*2)*head_a ## OK

            # H3
            # h = 0.3 * head_a + 0.7 * head_t
            # roll_scale = 25
            # roll_r = math.exp(-((a_r/roll_scale)**4))
            # roll_rr = math.exp(-((a_rr/roll_scale)**2))
            # # reward = h * (d+v) + head_a*((roll_r-0.5)*2 + (roll_rr-0.5)*2)
            # reward = h * (d+v) + head_a*((roll_r-0.5) + (roll_rr-0.5)*2) #H3r

            # H4 #### good
            # h = 0.35 * head_a + 0.65 * head_t
            # roll_scale = 25
            # roll_r = math.exp(-((a_r/roll_scale)**4))
            # roll_rr = math.exp(-((a_rr/(roll_scale*2))**2))
            # reward = h * (d+v) + h*((roll_r-0.5)*2 + (roll_rr-0.5)*2)

            # H5 basicly H4
            # h = 0.35 * head_a + 0.65 * head_t
            # roll_scale = 25
            # roll_r = math.exp(-((a_r/roll_scale)**4))
            # roll_rr = math.exp(-((a_rr/(roll_scale*2))**2))
            # reward = h * (d+v) + 0.5*h*((roll_r-0.5)*2 + (roll_rr-0.5)*2)

            # H6: GOOD
            # h = 0.35 * head_a + 0.65 * head_t
            # roll_scale = 40
            # roll_rr_scale = 30
            # roll_r = math.exp(-((a_r/roll_scale)**4))
            # roll_rr = math.exp(-((a_rr/roll_rr_scale)**2))
            # reward = h * (d+v) + h*(0.3*(roll_r-0.5)*2 + 0.7*(roll_rr-0.5)*2)


            # H7 # VERY GOOD, track_scale 0.1
            # h = 0.35 * head_a + 0.65 * head_t
            # roll_scale = 35
            # roll_rr_scale = 25
            # pitch_rr_scale = 15
            # roll_r = math.exp(-((a_r/roll_scale)**4))
            # roll_rr = math.exp(-((a_rr/roll_rr_scale)**2))
            # pitch_rr = math.exp(-((a_pr/pitch_rr_scale)**2))
            # reward = h * (d+v) + 0.7*h*(0.3*(roll_r-0.5)*2 + 0.6*(roll_rr-0.5)*2 + 0.1*(pitch_rr-0.5)*2)

            # H8
            h = 0.35 * head_a + 0.65 * head_t
            roll_scale = 35
            roll_rr_scale = 25
            pitch_rr_scale = 15
            roll_r = math.exp(-((a_r/roll_scale)**4))
            roll_rr = math.exp(-((a_rr/roll_rr_scale)**2))
            pitch_rr = math.exp(-((a_pr/pitch_rr_scale)**2))
            reward = h*(0.7*(d+v) + 0.3*(0.3*(roll_r-0.5)*2 + 0.5*(roll_rr-0.5)*2 + 0.2*(pitch_rr-0.5)*2))

        elif self.roll_only:
            head_scale = 40
            head_a = 1-math.exp(-((a_a/head_scale)**2))
            head_t = math.exp(-((a_t/head_scale)**2))

            roll_scale = 60
            roll_rr_scale = 25
            pitch_rr_scale = 20
            roll_r = math.exp(-((a_r/roll_scale)**2)) -1.0
            roll_rr = math.exp(-((a_rr/roll_rr_scale)**4)) - 0.5
            pitch_rr = math.exp(-((a_pr/pitch_rr_scale)**4)) - 0.5
            reward = (0.5*head_a+0.5*head_t)*(roll_rr + pitch_rr + 0.5*roll_r)

        else:
            head_scale = 30
            head_a = math.exp(-((a_a/head_scale)**2))
            head_t = math.exp(-((a_t/(head_scale))**2))

            # H9 - cannon
            # h = 0.6 * head_a + 0.4 * head_t
            # roll_rr_scale = 25 ##
            # pitch_rr_scale = 15 ##
            # roll_rr = math.exp(-((a_rr/roll_rr_scale)**2))
            # pitch_rr = math.exp(-((a_pr/pitch_rr_scale)**2))
            # reward = h*(0.5*(roll_rr-0.5) + 0.5*(pitch_rr-0.5))

            # H10 - cannon L1
            # h = 0.6 * head_a + 0.4 * head_t
            # roll_scale = 60
            # roll_rr_scale = 25
            # pitch_rr_scale = 20
            # roll_r = math.exp(-((a_r/roll_scale)**2)) -1.0
            # roll_rr = math.exp(-((a_rr/roll_rr_scale)**4)) - 0.5
            # pitch_rr = math.exp(-((a_pr/pitch_rr_scale)**4)) - 0.5
            # reward = h*(roll_rr + pitch_rr + 0.5*roll_r)

            # H11 - cannon L2
            h = 0.7 * head_a + 0.3 * head_t
            roll_scale = 60
            roll_rr_scale = 25
            pitch_rr_scale = 20
            roll_r = math.exp(-((a_r/roll_scale)**2)) -1.0
            roll_rr = math.exp(-((a_rr/roll_rr_scale)**4)) - 0.5
            pitch_rr = math.exp(-((a_pr/pitch_rr_scale)**4)) - 0.5
            reward = h*(roll_rr + pitch_rr + 0.5*roll_r)

        return reward
        
        # else:
        #     dist_scale = 7000
        #     if dist < 4000:
        #         dist_r = 1
        #     else:
        #         dist_r = math.exp(-(((dist-4000)/dist_scale)**4))

        #     head_scale = 20
        #     head_a = math.exp(-((a_a/head_scale)**2))

        #     roll_scale = 20
        #     roll_r = math.exp(-((a_r/roll_scale)**2))

        #     reward = (dist_r-1) * (0.5*(1-head_a) + 0.5*(1-roll_r))

        #     return reward