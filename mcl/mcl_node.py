import os
import random
import copy
import yaml

import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math
import matplotlib.patches as patches
import numpy as np
from scipy.stats import multivariate_normal

class World:
    def __init__(self, time_span, time_interval, debug = False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8,8))                # 8x8 inchの図を準備
        ax = fig.add_subplot(111)                      # サブプロットを準備
        ax.set_aspect('equal')                         # 縦横比を座標の値と一致させる
        ax.set_xlim(-5,5)                              # X軸を-5m x 5mの範囲で描画
        ax.set_ylim(-5,5)                              # Y軸も同様に
        ax.set_xlabel("X",fontsize=20)                 # X軸にラベルを表示
        ax.set_ylabel("Y",fontsize=20) 

        elems = []

        if self.debug:        
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)
        else:
            self.ani = ani.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                     frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        elems.append(ax.text(-4.4, 4.5, "t = "+str(i), fontsize = 10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(0.1)

class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight

    def motion_update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs() #順にnn, no, on, oo
        noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose = Particle.state_transition(noised_nu, noised_omega, time, self.pose)


    def state_transition(nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(t0), nu*math.sin(t0), omega]) * time
        else:
            return pose + np.array([nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), nu/omega*(-math.cos(t0 + omega * time) + math.cos(t0)), omega * time])

    def observation_update(self, observation, env_map, distance_dev_rate, direction_dev):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            pos_on_map = env_map.landmarks[obs_id].pos
            particle_suggest_pos = Camera.observation_function(self.pose, pos_on_map)

            distance_dev = distance_dev_rate*particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)

class Mcl:
    def __init__(self, env_map, init_pose, num, motion_noise_stds, \
                 distance_dev=0.14, direction_dev=0.05):
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]
        self.map = env_map
        self.distance_dev = distance_dev
        self.direction_dev = direction_dev

        v = motion_noise_stds
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)

    def motion_update(self, nu, omega, time):
        for p in self.particles: p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, color="blue", alpha=0.5))

    def observation_update(self, observation):
        for p in self.particles: p.observation_update(observation, self.map, self.distance_dev, self.direction_dev)
        self.resampling()

    def resampling(self):
        ws = np.cumsum([e.weight for e in self.particles])
        if ws[-1] < 1e-100: ws = [e + 1e-100 for e in ws]

        step = ws[-1]/len(self.particles)
        r = np.random.uniform(0.0, step)
        cur_pos = 0
        ps = []

        while(len(ps) < len(self.particles)):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1

        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles: p.weight = 1.0/len(self.particles)

class Camera:
    def __init__(self, env_map, \
                 distance_range=(0.5, 3.0),
                 direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos):
        if polarpos is None:
            return False
        
        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] \
            and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color="pink")

    @classmethod
    def observation_function(ls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]

        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi

        return np.array([np.hypot(*diff), phi]).T

class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation = None):
        return self.nu, self.omega

class EstimatorAgent(Agent):
    def __init__(self, time_interval, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0
        self.prev_omega = 0

    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)

class Robot:
    def __init__(self, pose, agent = None, sensor = None, color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.sensor = sensor
        self.agent = agent
        self.poses = [pose]

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x, xn], [y, yn], color = self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        elems.append(ax.add_patch(c))   # 上のpatches.Circleでロボットの胴体を示す円を作ってサブプロットへ登録
        self.poses.append(self.pose)

        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])

        if self.agent:
            self.agent.draw(ax, elems)

    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]

        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(t0), nu*math.sin(t0), omega]) * time
        else:
            return pose + np.array([nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), nu/omega*(-math.cos(t0 + omega * time) + math.cos(t0)), omega * time])

    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)

class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))

class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks: lm.draw(ax, elems)

def main():
    mcl_dir = os.path.dirname(os.path.abspath('mcl'))
    config_file_path = os.path.join(mcl_dir, 'config', 'params.yaml')

    with open(config_file_path, 'r') as file:
        config =yaml.safe_load(file)

    time_interval = config.get('time_interval', 0.0)
    initial_pose = config.get('initial_pose', 0.0)
    linear_vel = config.get('linear_vel', 0.0)
    linear_distance_std = config.get('linear_distance_std', 0.0)
    angular_distance_std = config.get('angular_distance_std', 0.0)
    linear_rotation_std = config.get('linear_rotation_std', 0.0)
    angular_rotation_std = config.get('angular_rotation_std', 0.0)        

    init_pose = np.array(initial_pose).T
    world = World(100, time_interval)

    m = Map()
    m.append_landmark(Landmark(2, 2))
    world.append(m)

    estimator = Mcl(m, init_pose, 100,
                    motion_noise_stds={
                        "nn":linear_distance_std,
                        "no":angular_distance_std,
                        "on":linear_rotation_std,
                        "oo":angular_rotation_std}
                    )

    straight = EstimatorAgent(time_interval, linear_vel, 0.0, estimator)
    robot = Robot(init_pose, sensor=Camera(m), agent=straight)
    estimator.motion_update(0.2, 0, time_interval)

    world.append(robot)
    world.draw()

if __name__ == "__main__":
    main()
