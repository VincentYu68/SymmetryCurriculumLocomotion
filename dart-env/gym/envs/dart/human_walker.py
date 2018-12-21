__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import joblib
import os

import time

import pydart2 as pydart


class DartHumanWalkerEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0] * 23, [-1.0] * 23])
        self.action_scale = np.array([60.0, 160, 60, 100, 80, 60, 60, 160, 60, 100, 80, 60, 150, 150, 100, 15,100,15, 30, 15,100,15, 30])
        self.action_scale *= 1.0
        self.action_penalty_weight = np.array([1.0]*23)#np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        obs_dim = 57

        self.t = 0
        self.target_vel = 0.0
        self.init_tv = 0.0
        self.final_tv = 1.5
        self.tv_endtime = 1.0
        self.tvel_diff_perc = 1.0
        self.smooth_tv_change = True
        self.running_average_velocity = False
        self.running_avg_rew_only = True
        self.avg_rew_weighting = []
        self.vel_cache = []
        self.init_pos = 0
        self.pos_spd = False # Use spd on position in forward direction. Only use when treadmill is used
        self.assist_timeout = 0.0  # do not provide pushing assistance after certain time
        self.assist_schedule = [[0.0, [2000, 2000]], [3.0, [1500, 1500.0]], [6.0, [1125, 1125]]]
        self.reset_range = 0.005

        self.rand_target_vel = False
        self.init_push = False
        self.enforce_target_vel = True
        self.const_force = None
        self.hard_enforce = False
        self.treadmill = False
        self.treadmill_vel = -self.init_tv
        self.treadmill_init_tv = -0.0
        self.treadmill_final_tv = -4.5
        self.treadmill_tv_endtime = 6.0

        self.base_policy = None
        self.push_target = 'pelvis'

        self.constrain_dcontrol = 1.0
        self.previous_control = None

        self.total_act_force = 0
        self.total_ass_force = 0

        self.energy_weight = 0.3
        self.alive_bonus_rew = 9.0

        self.cur_step = 0
        self.stepwise_rewards = []
        self.conseq_limit_pen = 0  # number of steps lying on the wall
        self.constrain_2d = True
        self.init_balance_pd = 6000.0
        self.init_vel_pd = 3000.0
        self.end_balance_pd = 6000.0
        self.end_vel_pd = 3000.0

        self.pd_vary_end = self.target_vel * 6.0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.local_spd_curriculum = True
        self.anchor_kp = np.array([2000.0, 1000.0])
        self.curriculum_step_size = 0.1  # 10%
        self.min_curriculum_step = 50  # include (0, 0) if distance between anchor point and origin is smaller than this value

        # state related
        self.contact_info = np.array([0, 0])
        self.contact_locations = [[], []]
        self.include_additional_info = True
        if self.include_additional_info:
            obs_dim += len(self.contact_info)
        if self.running_average_velocity or self.smooth_tv_change:
            obs_dim += 1

        self.curriculum_id = 0
        self.spd_kp_candidates = None

        self.vel_reward_weight = 3.0
        self.stride_weight = 0.0

        self.init_qs = []
        self.init_dqs = []

        if self.treadmill:
            dart_env.DartEnv.__init__(self, 'kima/kima_human_edited_treadmill.skel', 15, obs_dim, self.control_bounds,
                                      disableViewer=True, dt=0.002)
        else:
            dart_env.DartEnv.__init__(self, 'kima/kima_human_edited.skel', 15, obs_dim, self.control_bounds,
                                      disableViewer=True, dt=0.002)

        # add human joint limit
        '''skel = self.robot_skeleton
        world = self.dart_world
        leftarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(skel.joint('j_bicep_left'),
                                                                            skel.joint('j_forearm_left'), False)
        rightarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(skel.joint('j_bicep_right'),
                                                                             skel.joint('j_forearm_right'), True)
        leftlegConstraint = pydart.constraints.HumanLegJointLimitConstraint(skel.joint('j_thigh_left'),
                                                                            skel.joint('j_shin_left'),
                                                                            skel.joint('j_heel_left'), False)
        rightlegConstraint = pydart.constraints.HumanLegJointLimitConstraint(skel.joint('j_thigh_right'),
                                                                             skel.joint('j_shin_right'),
                                                                             skel.joint('j_heel_right'), True)
        leftarmConstraint.add_to_world(world)
        rightarmConstraint.add_to_world(world)
        leftlegConstraint.add_to_world(world)
        rightlegConstraint.add_to_world(world)'''

        self.robot_skeleton.set_self_collision_check(True)

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(20)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(20)

        # self.dart_world.set_collision_detector(3)

        self.sim_dt = self.dt / self.frame_skip

        for bn in self.robot_skeleton.bodynodes:
            if len(bn.shapenodes) > 0:
                if hasattr(bn.shapenodes[0].shape, 'size'):
                    shapesize = bn.shapenodes[0].shape.size()
                    print('density of ', bn.name, ' is ', bn.mass()/np.prod(shapesize))
        print('Total mass: ', self.robot_skeleton.mass())

        self.use_ref_policy = False
        self.ref_policy = None
        self.ref_policy_curriculum = np.array([2000, 1000])
        self.ref_strength_q = 0.005
        self.ref_strength_dq = 0.0005
        self.ref_feat_strength = 0.25
        self.ref_dfeat_strength = 0.025
        self.ref_trajs = []
        self.ref_features = []
        self.chosen_traj = [0, 0]
        self.max_eps_step_ref = 100 # maximum episode step when using reference policy
        self.ref_traj_num = 10

        utils.EzPickle.__init__(self)

    # only 1d
    def _spd(self, target_q, id, kp, target_dq=None):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_dq is not None:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (self.robot_skeleton.M[id][id] + self.Kd * self.sim_dt)
        if target_dq is None:
            p = -self.Kp * (self.robot_skeleton.q[id] + self.robot_skeleton.dq[id] * self.sim_dt - target_q[id])
        else:
            p = 0
        d = -self.Kd * (self.robot_skeleton.dq[id] - target_dq)
        qddot = invM * (-self.robot_skeleton.c[id] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt

        return tau

    def _bodynode_spd(self, bn, kp, dof, target_vel=None):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_vel is not None:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (bn.mass() + self.Kd * self.sim_dt)
        p = -self.Kp * (bn.C[dof] + bn.dC[dof] * self.sim_dt)
        if target_vel is None:
            target_vel = 0.0
        d = -self.Kd * (bn.dC[dof] - target_vel * 1.0)  # compensate for average velocity match
        qddot = invM * (-bn.C[dof] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt

        return tau

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            force = 0
            if self.constrain_2d and (self.t < self.assist_timeout):
                force = self._bodynode_spd(self.robot_skeleton.bodynode(self.push_target), self.current_pd, 2)
                self.robot_skeleton.bodynode(self.push_target).add_ext_force(np.array([0, 0, force]))

            if self.enforce_target_vel and (self.t < self.assist_timeout) and not self.hard_enforce:
                tvel = self.target_vel
                if self.treadmill:
                    tvel += self.treadmill_vel
                if self.const_force is None:
                    force = self._bodynode_spd(self.robot_skeleton.bodynode(self.push_target), self.vel_enforce_kp, 0,
                                           tvel * self.tvel_diff_perc)
                else:
                    if self.robot_skeleton.bodynode(self.push_target).dC[0] < tvel*0.5:
                        force = self.const_force
                    else:
                        force = 0
                self.push_forces.append(force)
                self.robot_skeleton.bodynode(self.push_target).add_ext_force(np.array([force, 0, 0]))
            self.total_act_force += np.linalg.norm(tau)

            self.total_ass_force += np.abs(force)

            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()
            s = self.state_vector()
            if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
                break
        #print(self.total_act_force, self.total_ass_force)

    def advance(self, clamped_control):
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        if self.enforce_target_vel:
            if self.hard_enforce and self.treadmill:
                current_dq_tread = self.dart_world.skeletons[0].dq
                current_dq_tread[0] = self.treadmill_vel  # * np.min([self.t/4.0, 1.0])
                self.dart_world.skeletons[0].dq = current_dq_tread
            elif self.hard_enforce:
                current_dq = self.robot_skeleton.dq
                current_dq[0] = self.target_vel
                self.robot_skeleton.dq = current_dq
        self.do_simulation(tau, self.frame_skip)

    def clamp_act(self, a):
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
        return clamped_control

    def _step(self, a):
        if self.use_ref_policy:
            current_obs = self._get_obs()

        # smoothly increase the target velocity
        if self.smooth_tv_change:
            self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (
                    self.final_tv - self.init_tv) + self.init_tv
            self.treadmill_vel = (np.min([self.t, self.treadmill_tv_endtime]) / self.treadmill_tv_endtime) * (
                    self.treadmill_final_tv - self.treadmill_init_tv) + self.treadmill_init_tv

        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        if len(self.assist_schedule) > 0:
            for sch in self.assist_schedule:
                if self.t > sch[0]:
                    self.current_pd = sch[1][0]
                    self.vel_enforce_kp = sch[1][1]

        posbefore = self.robot_skeleton.bodynode(self.push_target).com()[0]

        clamped_control = self.clamp_act(a)

        self.advance(np.copy(clamped_control))

        posafter = self.robot_skeleton.bodynode(self.push_target).com()[0]
        height = self.robot_skeleton.bodynode('head').com()[1]
        side_deviation = self.robot_skeleton.bodynode('head').com()[2]
        angle = self.robot_skeleton.q[3]

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynode('head').to_world(
            np.array([0, 1, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynode('head').to_world(
            np.array([1, 0, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

        lateral = np.array([0, 0, 1])
        lateral_world = self.robot_skeleton.bodynode('head').to_world(
            np.array([0, 0, 1])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        lateral_world /= np.linalg.norm(lateral_world)
        ang_cos_ltl = np.dot(lateral, lateral_world)
        ang_cos_ltl = np.arccos(ang_cos_ltl)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        self.contact_info = np.array([0, 0])
        stride_reward = 0.0
        l_foot_force = np.array([0.0, 0, 0])
        r_foot_force = np.array([0.0, 0, 0])
        in_contact = False
        for contact in contacts:
            if contact.skel_id1 == contact.skel_id2:
                self_colliding = True
            if contact.skel_id1 + contact.skel_id2 == 1:
                if contact.bodynode1 == self.robot_skeleton.bodynode(
                        'l-foot') or contact.bodynode2 == self.robot_skeleton.bodynode('l-foot'):
                    self.contact_info[0] = 1
                    total_force_mag += np.linalg.norm(contact.force)
                    l_foot_force += contact.force
                    in_contact = True
                    if self.t > self.tv_endtime:
                        if len(self.contact_locations[0]) > 0 and np.linalg.norm(self.contact_locations[0][-1] - self.robot_skeleton.bodynode('l-foot').C) > 0.2:
                            self.contact_locations[0].append(self.robot_skeleton.bodynode('l-foot').C)
                            stride_reward = self.stride_weight*(- np.abs(
                                np.linalg.norm(self.contact_locations[0][-1] - self.contact_locations[0][-2]) - 1.2))
                        if len(self.contact_locations[0]) == 0:
                            self.contact_locations[0].append(self.robot_skeleton.bodynode('l-foot').C)
                if contact.bodynode1 == self.robot_skeleton.bodynode(
                        'r-foot') or contact.bodynode2 == self.robot_skeleton.bodynode('r-foot'):
                    self.contact_info[1] = 1
                    total_force_mag += np.linalg.norm(contact.force)
                    r_foot_force += contact.force
                    in_contact = True
                    if self.t > self.tv_endtime:
                        if len(self.contact_locations[1]) > 0 and np.linalg.norm(self.contact_locations[1][-1] - self.robot_skeleton.bodynode('r-foot').C) > 0.2:
                            self.contact_locations[1].append(self.robot_skeleton.bodynode('r-foot').C)
                            stride_reward = self.stride_weight * (- np.abs(
                                np.linalg.norm(self.contact_locations[1][-1] - self.contact_locations[1][-2]) - 1.2))

                        if len(self.contact_locations[1]) == 0:
                            self.contact_locations[1].append(self.robot_skeleton.bodynode('r-foot').C)
        if in_contact:
            self.enforce_target_vel = True
        else:
            self.enforce_target_vel = True

        alive_bonus = self.alive_bonus_rew#np.max([1.5 + self.final_tv * 0.5 * self.vel_reward_weight, 4.0])

        vel = (posafter - posbefore) / self.dt
        vel_rew = 0.0
        if self.pos_spd:
            self.vel_cache.append(posafter) # actually position cache
        else:
            self.vel_cache.append(vel)
            self.target_vel_cache.append(self.target_vel)
        if self.running_average_velocity or self.running_avg_rew_only:
            if self.t < self.tv_endtime:
                self.avg_rew_weighting.append(1.0)
            else:
                self.avg_rew_weighting.append(1)

        vel_rew_scale = 1.0
        if len(self.vel_cache) > int(2.0/self.dt) and (self.running_average_velocity or self.running_avg_rew_only):
            self.vel_cache.pop(0)
            self.avg_rew_weighting.pop(0)
            self.target_vel_cache.pop(0)
        else:
            vel_rew_scale = 1.0#np.min([len(self.vel_cache) * self.dt, 1.0])

        if not self.treadmill:
            if self.reference_trajectory is not None:
                vel_rew = - np.exp(3.0*(np.abs(self.reference_trajectory[self.cur_step][0] - self.robot_skeleton.com()[0]**2)))
            elif self.running_average_velocity or self.running_avg_rew_only:
                vel_rew = -self.vel_reward_weight * np.abs(np.mean(self.target_vel_cache) - np.mean(self.vel_cache))
            else:
                vel_diff = np.abs(self.target_vel - vel)
                vel_rew = -3.0 * vel_diff
        else:
            if self.running_average_velocity or self.running_avg_rew_only:
                append_vel = np.ones(int(1.0/self.dt) - len(self.vel_cache)) * (self.target_vel + self.treadmill_vel)
                vel_rew = -3.0 *vel_rew_scale* (np.abs(self.target_vel + self.treadmill_vel - np.mean(np.append(self.vel_cache, append_vel))))
            else:
                vel_rew = -3.0 * (np.abs(self.target_vel + self.treadmill_vel - vel))
        if self.t < self.tv_endtime:
            vel_rew *= 0.5
        # vel_rew *= 0
        # action_pen = 5e-1 * (np.square(a)* actuator_pen_multiplier).sum()
        action_pen = self.energy_weight * np.abs(a * self.action_penalty_weight).sum()# * (1.5/np.max([2.0,self.target_vel]))
        # action_pen += 0.02 * np.sum(np.abs(a* self.robot_skeleton.dq[6:]))
        deviation_pen = 3 * abs(side_deviation)

        contact_pen = 0.5 * np.square(np.clip(l_foot_force, -2000, 2000)/ 1000.0).sum()+np.square(np.clip(r_foot_force, -2000, 2000)/ 1000.0).sum()

        rot_pen = 0.3 * (abs(ang_cos_uwd)) + 0.3 * (abs(ang_cos_fwd)) + 1.5 * (abs(ang_cos_ltl))
        # penalize bending of spine
        spine_pen = 1.0 * np.sum(np.abs(self.robot_skeleton.q[[18, 19]])) + 1.5 * np.abs(self.robot_skeleton.q[20]) + 0.8 * np.abs(self.robot_skeleton.q[19] + self.robot_skeleton.q[3])

        #spine_pen += 0.05 * np.sum(np.abs(self.robot_skeleton.q[[8, 14]]))

        dq_pen = 0.0 * np.sum(np.square(self.robot_skeleton.dq[6:]))


        #torso_vel_pen = 0.15*np.abs(self.robot_skeleton.bodynode('thorax').com_spatial_velocity()[0:3]).sum()
        reward = vel_rew + alive_bonus - action_pen - deviation_pen - rot_pen*0 - spine_pen - dq_pen# - contact_pen #+ stride_reward # - torso_vel_pen
        pos_rew = vel_rew + alive_bonus - deviation_pen - rot_pen - spine_pen
        neg_pen = - action_pen

        ref_reward = 0
        ref_feat_rew = 0
        self.t += self.dt

        self.cur_step += 1

        s = self.state_vector()
        if self.use_ref_policy:
            #reward -= self.ref_strength_q * np.abs(self.robot_skeleton.q - target_q).sum()
            #reward -= self.ref_strength_dq * np.abs(self.robot_skeleton.dq - target_dq).sum()
            #ref_reward = self.ref_strength_q * np.abs(self.robot_skeleton.q - target_q).sum() + self.ref_strength_dq * np.abs(self.robot_skeleton.dq - target_dq).sum()
            '''cur_poses = [self.robot_skeleton.bodynode('l-lowerarm').C,
                            self.robot_skeleton.bodynode('r-lowerarm').C, self.robot_skeleton.bodynode('l-foot').C,
                            self.robot_skeleton.bodynode('r-foot').C]
            cur_dposes = [self.robot_skeleton.bodynode('l-lowerarm').dC,
                             self.robot_skeleton.bodynode('r-lowerarm').C, self.robot_skeleton.bodynode('l-foot').dC,
                             self.robot_skeleton.bodynode('r-foot').dC]
            for p in range(len(cur_poses)):
                ref_reward += self.ref_strength_q * np.abs(cur_poses[p] - target_poses[p]).sum()
                ref_reward += self.ref_strength_dq * np.abs(cur_dposes[p] - target_dposes[p]).sum()
            reward += - ref_reward'''
            ref_state_vec = self.ref_trajs[self.chosen_traj[0]][self.chosen_traj[1] + self.cur_step]
            weight = np.exp(- 0.04*(self.cur_step-1))
            ref_reward -= self.ref_strength_q * weight * np.sum(np.abs(self.state_vector() - ref_state_vec)[0:int(len(self.robot_skeleton.q)/2)])
            ref_reward -= self.ref_strength_dq * weight * np.sum(
                np.abs(self.state_vector() - ref_state_vec)[int(len(self.robot_skeleton.q) / 2):])
            for k in self.ref_features[self.chosen_traj[0]][self.chosen_traj[1] + self.cur_step].keys():
                ref_feat_rew -= self.ref_strength_q * weight * np.sum(np.square(self.robot_skeleton.bodynode(k).C - self.ref_features[self.chosen_traj[0]][self.chosen_traj[1] + self.cur_step][k][0]))
                ref_feat_rew -= self.ref_strength_dq * weight * np.sum(np.square(self.robot_skeleton.bodynode(k).dC - self.ref_features[self.chosen_traj[0]][self.chosen_traj[1] + self.cur_step][k][1]))
            reward += 1.0 + ref_reward + ref_feat_rew


        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height - self.init_height > -0.35) and (height - self.init_height < 1.0) and (
                    abs(ang_cos_uwd) < 1.2) and (abs(ang_cos_fwd) < 1.2)
                    and np.abs(angle) < 1.2 and
                    np.abs(self.robot_skeleton.q[5]) < 1.2 and np.abs(self.robot_skeleton.q[4]) < 1.2 and np.abs(self.robot_skeleton.q[3]) < 1.2
                    and np.abs(side_deviation) < 0.9)

        if self.use_ref_policy and self.cur_step > self.max_eps_step_ref:
            done = True

        self.stepwise_rewards.append(reward)

        self.previous_control = clamped_control

        ob = self._get_obs()

        broke_sim = False
        if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
            broke_sim = True

        '''work = np.sum(np.abs(clamped_control * self.action_scale * self.robot_skeleton.dq[6:])) * self.dt
        self.total_work += work
        cot = self.total_work / (self.robot_skeleton.mass() * 9.81 * self.robot_skeleton.C[0])
        print(cot)'''

        return ob, reward, done, {'broke_sim': broke_sim, 'vel_rew': vel_rew, 'action_pen': action_pen,
                                  'deviation_pen': deviation_pen, 'curriculum_id': self.curriculum_id,
                                  'curriculum_candidates': self.spd_kp_candidates, 'done_return': done,
                                  'dyn_model_id': 0, 'state_index': 0, 'com': self.robot_skeleton.com(),
                                  'pos_rew': pos_rew, 'neg_pen': neg_pen, 'contact_locations':self.contact_locations,
                                  'contact_force': l_foot_force + r_foot_force,
                                  'contact_forces':[l_foot_force,r_foot_force],
                                  'ref_reward': ref_reward, 'ref_feat_rew':ref_feat_rew, 'avg_vel':np.mean(self.vel_cache),
                                  }

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        if self.include_additional_info:
            state = np.concatenate([state, self.contact_info])

        if self.rand_target_vel or self.smooth_tv_change:
            state = np.concatenate([state, [self.target_vel]])

        if self.running_average_velocity:
            state = np.concatenate([state, [(self.robot_skeleton.q[0] - self.init_pos) / self.t]])

        return state

    def reset_model(self):
        #print('resetttt')
        self.dart_world.reset()
        self.total_work = 0

        init_q = self.robot_skeleton.q
        init_dq = self.robot_skeleton.dq
        if len(self.init_dqs) > 0:
            init_pid = np.random.randint(len(self.init_qs))
            init_q = self.init_qs[init_pid]
            init_dq = self.init_dqs[init_pid]

        qpos = init_q + self.np_random.uniform(low=-self.reset_range, high=self.reset_range, size=self.robot_skeleton.ndofs)
        qvel = init_dq + self.np_random.uniform(low=-self.reset_range*10, high=self.reset_range*10, size=self.robot_skeleton.ndofs)
        s = np.sign(np.random.random()-0.5) * self.np_random.uniform(low=.005, high=.05)
        #qpos[1] += s
        #qpos[7] -= s

        if self.rand_target_vel:
            self.target_vel = np.random.uniform(0.8, 2.5)

        if self.local_spd_curriculum:
            self.spd_kp_candidates = [self.anchor_kp]
            self.curriculum_id = np.random.randint(len(self.spd_kp_candidates))
            chosen_curriculum = self.spd_kp_candidates[self.curriculum_id]
            self.init_balance_pd = chosen_curriculum[0]
            self.end_balance_pd = chosen_curriculum[0]
            self.init_vel_pd = chosen_curriculum[1]
            self.end_vel_pd = chosen_curriculum[1]

        if self.init_push:
            qvel[0] = self.target_vel
        self.set_state(qpos, qvel)
        self.t = self.dt
        self.cur_step = 0
        self.stepwise_rewards = []

        self.init_pos = self.robot_skeleton.q[0]

        self.conseq_limit_pen = 0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd
        if self.const_force is not None:
            self.const_force = self.init_vel_pd

        self.contact_info = np.array([0, 0])
        self.contact_locations = [[], []]

        self.push_forces = []

        self.previous_control = None

        self.init_height = self.robot_skeleton.bodynode('head').C[1]
        self.moving_bin = None

        self.vel_cache = []
        self.target_vel_cache = []

        self.avg_rew_weighting = []

        self.reference_trajectory = None

        if np.random.randint(2) == 0:
            self.push_target = 'pelvis'
        #else:
        #    self.push_target = 'thorax'

        if self.use_ref_policy:
            if len(self.ref_trajs) == 0:
                self.use_ref_policy = False
                current_pd = self.current_pd
                vel_enforce_kp = self.vel_enforce_kp

                for i in range(50):
                    one_traj = []
                    one_traj_feature = []
                    o = self.reset()
                    self.init_balance_pd = self.ref_policy_curriculum[0]
                    self.init_vel_pd = self.ref_policy_curriculum[1]
                    one_traj.append(self.state_vector())
                    one_traj_feature.append({'l-foot': [self.robot_skeleton.bodynode('l-foot').C,
                                                        self.robot_skeleton.bodynode('l-foot').dC],
                                             'r-foot': [self.robot_skeleton.bodynode('r-foot').C,
                                                        self.robot_skeleton.bodynode('r-foot').dC],
                                             'l-lowerarm': [self.robot_skeleton.bodynode('l-lowerarm').C,
                                                            self.robot_skeleton.bodynode('l-lowerarm').dC],
                                             'r-lowerarm': [self.robot_skeleton.bodynode('r-lowerarm').C,
                                                            self.robot_skeleton.bodynode('r-lowerarm').dC]})
                    for j in range(400):
                        a, v = self.ref_policy.act(False, o)
                        o, r, d, _ = self._step(a)
                        one_traj.append(self.state_vector())
                        one_traj_feature.append({'l-foot':[self.robot_skeleton.bodynode('l-foot').C,self.robot_skeleton.bodynode('l-foot').dC],
                                                 'r-foot':[self.robot_skeleton.bodynode('r-foot').C,self.robot_skeleton.bodynode('r-foot').dC],
                                                 'l-lowerarm':[self.robot_skeleton.bodynode('l-lowerarm').C, self.robot_skeleton.bodynode('l-lowerarm').dC],
                                                 'r-lowerarm':[self.robot_skeleton.bodynode('r-lowerarm').C, self.robot_skeleton.bodynode('r-lowerarm').dC]})
                        if d:
                            break
                    if len(one_traj) > 300:
                        self.ref_trajs.append(one_traj)
                        self.ref_features.append(one_traj_feature)
                    if len(self.ref_trajs) > self.ref_traj_num:
                        break
                self.current_pd = current_pd
                self.vel_enforce_kp = vel_enforce_kp
                self.use_ref_policy = True
                self.reset()
            self.chosen_traj[0] = np.random.randint(len(self.ref_trajs))
            select_from_transit = np.random.randint(2) == 0
            if select_from_transit:
                self.chosen_traj[1] = np.random.randint(int(self.tv_endtime / self.dt))
            else:
                self.chosen_traj[1] = np.random.randint(int(self.tv_endtime / self.dt), len(self.ref_trajs[self.chosen_traj[0]]) - self.max_eps_step_ref-1)
            self.set_state_vector(self.ref_trajs[self.chosen_traj[0]][self.chosen_traj[1]])
            self.t = self.chosen_traj[1] * self.dt

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            # self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[2] = -5.5

