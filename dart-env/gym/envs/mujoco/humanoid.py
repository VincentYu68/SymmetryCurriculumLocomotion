import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.t = 0
        self.push_timeout = 110.0  # do not provide pushing assistance after certain time
        self.assist_schedule = [[0.0, [2000, 2000]], [4.0, [1500, 1500.0]], [7.0, [1125, 1125]]]

        self.target_vel = 0
        self.init_tv = 0.0
        self.final_tv = 1.0
        self.tv_endtime = 1.0

        self.init_balance_pd = 0
        self.init_vel_pd = 0

        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)

        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               #data.cinert.flat,
                               #data.cvel.flat,
                               #data.qfrc_actuator.flat,
                               #data.cfrc_ext.flat,
                               [self.target_vel],
                               [self.t]])



    def virtual_assistant(self, dof, target_vel=None):
        mass = 8.32207894 # torso mass
        pos = self.model.data.qpos[0:3]
        vel  = self.model.data.qvel[0:3]
        self.Kp = self.current_pd
        self.Kd = self.Kp * self.model.opt.timestep
        if target_vel is not None:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (mass + self.Kd * self.model.opt.timestep)
        p = -self.Kp * (pos[dof] + vel[dof] * self.model.opt.timestep)
        if target_vel is None:
            target_vel = 0.0
        d = -self.Kd * (vel[dof] - target_vel * 1.0)  # compensate for average velocity match
        qddot = invM * (-pos[dof] + p + d)
        tau = p + d - self.Kd * (qddot) * self.model.opt.timestep

        return tau

    def do_simulation(self, ctrl, n_frames):
        self.model.data.ctrl = ctrl
        for _ in range(n_frames):
            if (self.t < self.push_timeout):
                force = self.virtual_assistant(1)
                xfrc_force = [0.0] * (6*self.model.nbody)
                xfrc_force[7] = force

                force = self.virtual_assistant(0, self.target_vel)
                xfrc_force[6] = force
                self.data.xfrc_applied = xfrc_force
            self.model.step()


    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.t = 0
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
