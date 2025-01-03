import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math
import matplotlib.patches as patches
import numpy as np

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
            if hasattr(obj, "one_step"): obj.one_step(1.0)

class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose

class Mcl:
    def __init__(self, init_pose, num):
        self.particles = [Particle(init_pose) for i in range(num)]

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, color="blue", alpha=0.5))

class Agent:
    def __init__(self, nu, omega, estimator):
        self.nu = nu
        self.omega = omega
        self.estimator = estimator

    def decision(self, observation = None):
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

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x, xn], [y, yn], color = self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        elems.append(ax.add_patch(c))   # 上のpatches.Circleでロボットの胴体を示す円を作ってサブプロットへ登録

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
        nu, omega = self.agent.decision()
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

class EstimationAgent(Agent):  ###EstimationAgent3 (1,2,6,7行目を記載)
    def __init__(self, nu, omega, estimator): 
        super().__init__(nu, omega)
        self.estimator = estimator
        
    def draw(self, ax, elems):   #追加
        self.estimator.draw(ax, elems)

def main():
    init_pose = np.array([-2, 0, 0]).T

    world = World(100, 0.1)
    m = Map()
    m.append_landmark(Landmark(3, 2))
    world.append(m)

    estimator = Mcl(init_pose, 100)

    straight = Agent(0.2, 0.0, estimator)
    robot = Robot(init_pose, agent=straight)

    world.append(robot)
    world.draw()

if __name__ == "__main__":
    main()
