__author__ = 'yuwenhao'


import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartDogRobotEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0] * 16, [-1.0] * 16])
        self.action_scale = np.array([80,200,160,30, 80,200,160,30, 40.0, 170, 160, 30, 40,170,160, 30])
        self.action_scale *= 1.0

        obs_dim = 43

        self.t = 0
        self.target_vel = 1.0
        self.init_tv = 0.0
        self.final_tv = 7.0
        self.tv_endtime = 3.0
        self.smooth_tv_change = True
        self.vel_cache = []
        self.init_pos = 0
        self.freefloat = 0.0
        self.assist_timeout = 0.0
        self.assist_schedule = [[0.0, [2000, 2000]], [3.0, [1500, 1500.0]], [6.0, [1125, 1125]]]

        self.init_push = False

        self.enforce_target_vel = True
        self.treadmill = False
        self.treadmill_vel = -self.init_tv
        self.treadmill_init_tv = -0.0
        self.treadmill_final_tv = -2.5
        self.treadmill_tv_endtime = 1.0

        self.running_avg_rew_only = True

        self.cur_step = 0
        self.stepwise_rewards = []
        self.constrain_2d = True
        self.init_balance_pd = 2000.0
        self.init_vel_pd = 2000.0
        self.end_balance_pd = 2000.0
        self.end_vel_pd = 2000.0

        self.constrain_dcontrol = 1.0
        self.previous_control = None

        self.energy_weight = 0.35
        self.alive_bonus = 11.0

        self.pd_vary_end = self.target_vel * 6.0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.local_spd_curriculum = True
        self.anchor_kp = np.array([2000, 1000])

        # state related
        self.contact_info = np.array([0, 0, 0, 0])

        if self.smooth_tv_change:
            obs_dim += 1
        self.include_additional_info = True
        if self.include_additional_info:
            obs_dim += len(self.contact_info)

        self.curriculum_id = 0
        self.spd_kp_candidates = None

        self.vel_reward_weight = 3.0

        dart_env.DartEnv.__init__(self, 'dog/dog_robot.skel', 15, obs_dim, self.control_bounds,
                                  disableViewer=True, dt=0.002)

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(10.0)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(10.0)

        self.sim_dt = self.dt / self.frame_skip

        for bn in self.robot_skeleton.bodynodes:
            if len(bn.shapenodes) > 0:
                if hasattr(bn.shapenodes[0].shape, 'size'):
                    shapesize = bn.shapenodes[0].shape.size()
                    print('density of ', bn.name, ' is ', bn.mass()/np.prod(shapesize))

        utils.EzPickle.__init__(self)

    # only 1d
    def _spd(self, target_q, id, kp, target_dq=0.0):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_dq > 0:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (self.robot_skeleton.M[id][id] + self.Kd * self.sim_dt)
        if target_dq == 0:
            p = -self.Kp * (self.robot_skeleton.q[id] + self.robot_skeleton.dq[id] * self.sim_dt - target_q[id])
        else:
            p = 0
        d = -self.Kd * (self.robot_skeleton.dq[id] - target_dq)
        qddot = invM * (-self.robot_skeleton.c[id] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt

        return tau

    def _bodynode_spd(self, bn, kp, dof, target_vel=0.0):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_vel > 0:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (bn.mass() + self.Kd * self.sim_dt)
        p = -self.Kp * (bn.C[dof] + bn.dC[dof] * self.sim_dt)
        d = -self.Kd * (bn.dC[dof] - target_vel)
        qddot = invM * (-bn.C[dof] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt

        return tau

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            if self.constrain_2d and self.t < self.assist_timeout:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('h_torso'), self.current_pd, 2)
                self.robot_skeleton.bodynode('h_torso').add_ext_force(np.array([0, 0, force]))

            if self.enforce_target_vel and self.t < self.assist_timeout:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('h_torso'), self.vel_enforce_kp, 0, self.target_vel)
                #if force < 0.0:
                #    force = 0.0
                self.robot_skeleton.bodynode('h_torso').add_ext_force(np.array([force, 0, 0]))
            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()
            s = self.state_vector()
            if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
                break

    def advance(self, a):
        #a[[3,4,5]]=a[[0,1,2]]
        #a[[9,10,11]]=a[[6,7,8]]
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if self.previous_control is not None:
                if clamped_control[i] > self.previous_control[i] + self.constrain_dcontrol:
                    clamped_control[i] = self.previous_control[i] + self.constrain_dcontrol
                elif clamped_control[i] < self.previous_control[i] - self.constrain_dcontrol:
                    clamped_control[i] = self.previous_control[i] - self.constrain_dcontrol
        self.previous_control = clamped_control

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        if len(self.assist_schedule) > 0:
            for sch in self.assist_schedule:
                if self.t > sch[0]:
                    self.current_pd = sch[1][0]
                    self.vel_enforce_kp = sch[1][1]

        # smoothly increase the target velocity
        if self.smooth_tv_change:
            self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (
                    self.final_tv - self.init_tv) + self.init_tv

        if self.t < self.freefloat:
            self.dart_world.set_gravity(np.array([0, -(self.t / self.freefloat) * 9.8, 0]))
        else:
            self.dart_world.set_gravity(np.array([0, -9.8, 0]))
        posbefore = self.robot_skeleton.bodynode('h_torso').com()[0]
        self.advance(np.copy(a))
        posafter = self.robot_skeleton.bodynode('h_torso').com()[0]
        height = self.robot_skeleton.bodynode('h_head').com()[1]
        side_deviation = self.robot_skeleton.bodynode('h_torso').com()[2]
        angle = self.robot_skeleton.q[3]

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynode('h_torso').to_world(
            np.array([0, 1, 0])) - self.robot_skeleton.bodynode('h_torso').to_world(np.array([0, 0, 0]))

        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynode('h_torso').to_world(
            np.array([1, 0, 0])) - self.robot_skeleton.bodynode('h_torso').to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

        lateral = np.array([0, 0, 1])
        lateral_world = self.robot_skeleton.bodynode('h_torso').to_world(
            np.array([0, 0, 1])) - self.robot_skeleton.bodynode('h_torso').to_world(np.array([0, 0, 0]))
        lateral_world /= np.linalg.norm(lateral_world)
        ang_cos_ltl = np.dot(lateral, lateral_world)
        ang_cos_ltl = np.arccos(ang_cos_ltl)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        self.contact_info = np.array([0, 0, 0, 0])
        body_hit_ground = False
        in_contact = False
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.skel_id1 == contact.skel_id2:
                self_colliding = True
            if contact.skel_id1 + contact.skel_id2 == 1:
                if contact.bodynode1 == self.robot_skeleton.bodynode(
                        'rear_foot_left') or contact.bodynode2 == self.robot_skeleton.bodynode('rear_foot'):
                    self.contact_info[0] = 1
                elif contact.bodynode1 == self.robot_skeleton.bodynode(
                        'rear_foot_left') or contact.bodynode2 == self.robot_skeleton.bodynode('rear_foot_left'):
                    self.contact_info[1] = 1
                elif contact.bodynode1 == self.robot_skeleton.bodynode(
                        'front_foot') or contact.bodynode2 == self.robot_skeleton.bodynode('front_foot'):
                    self.contact_info[2] = 1
                elif contact.bodynode1 == self.robot_skeleton.bodynode(
                        'front_foot_left') or contact.bodynode2 == self.robot_skeleton.bodynode('front_foot_left'):
                    self.contact_info[3] = 1
                else:
                    body_hit_ground = True
                in_contact = True
        if in_contact:
            self.enforce_target_vel = True
        else:
            self.enforce_target_vel = True


        vel = (posafter - posbefore) / self.dt

        self.vel_cache.append(vel)
        self.target_vel_cache.append(self.target_vel)
        if len(self.vel_cache) > int(1.0 / self.dt):
            self.vel_cache.pop(0)
            self.target_vel_cache.pop(0)

        vel_diff = np.abs(self.target_vel - vel)
        vel_rew = - 0.2 * self.vel_reward_weight * vel_diff
        if self.running_avg_rew_only:
            vel_rew = - self.vel_reward_weight * np.abs(np.mean(self.target_vel_cache) - np.mean(self.vel_cache))
        if self.t < self.tv_endtime:
            vel_rew *= 0.5

        action_pen = self.energy_weight * np.abs(a).sum()
        deviation_pen = 3 * abs(side_deviation)

        rot_pen = 1.0 * abs(self.robot_skeleton.q[3]) + 0.5 * abs(self.robot_skeleton.q[4]) + 1.5 * abs(self.robot_skeleton.q[5])
        reward = vel_rew + self.alive_bonus - action_pen - deviation_pen - rot_pen

        self.t += self.dt
        self.cur_step += 1

        s = self.state_vector()

        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height - self.init_height > -0.35) and (height - self.init_height < 1.0) and (
                    abs(ang_cos_uwd) < 1.2) and (abs(ang_cos_fwd) < 1.2)
                    and np.abs(angle) < 1.3
                    and np.abs(self.robot_skeleton.q[3]) < 1.2 and np.abs(self.robot_skeleton.q[4]) < 1.2 and np.abs(self.robot_skeleton.q[5]) < 1.2
                    and np.abs(side_deviation) < 0.9 and not body_hit_ground)

        self.stepwise_rewards.append(reward)

        ob = self._get_obs()

        broke_sim = False
        if not (np.isfinite(s).all()  and (np.abs(s[2:]) < 100).all()):
            broke_sim = True

        return ob, reward, done, {'broke_sim': broke_sim, 'vel_rew': vel_rew, 'action_pen': action_pen,
                                  'deviation_pen': deviation_pen, 'curriculum_id': self.curriculum_id,
                                  'curriculum_candidates': self.spd_kp_candidates, 'done_return': done,
                                  'dyn_model_id': 0, 'state_index': 0, 'avg_vel':np.mean(self.vel_cache)}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        if self.include_additional_info:
            state = np.concatenate([state, self.contact_info])
        if self.smooth_tv_change:
            state = np.concatenate([state, [self.target_vel]])

        return state

    def reset_model(self):
        self.dart_world.reset()
        self.init_height = self.robot_skeleton.bodynode('h_head').C[1]

        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)

        self.target_vel = self.init_tv
        if self.init_push:
            qvel[0] = self.target_vel

        if self.local_spd_curriculum:
            self.spd_kp_candidates = [self.anchor_kp]
            self.curriculum_id = np.random.randint(len(self.spd_kp_candidates))
            chosen_curriculum = self.spd_kp_candidates[self.curriculum_id]
            self.init_balance_pd = chosen_curriculum[0]
            self.end_balance_pd = chosen_curriculum[0]
            self.init_vel_pd = chosen_curriculum[1]
            self.end_vel_pd = chosen_curriculum[1]

        self.set_state(qpos, qvel)
        self.t = self.dt
        self.cur_step = 0
        self.stepwise_rewards = []

        self.init_pos = self.robot_skeleton.q[0]

        self.conseq_limit_pen = 0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.contact_info = np.array([0, 0, 0, 0])

        self.vel_cache = []
        self.target_vel_cache = []

        self.previous_control = None

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5