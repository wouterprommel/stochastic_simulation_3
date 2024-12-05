import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class sim():

    def __init__(self, n):
        self.n_particles = n
        self.particles = {}
        for i in range(self.n_particles):
            self.particles[i] = particle(i)

        self.step_size = 0.1 
        # part of annealing proces
        self.T = 200
        self.i_step = 0

    def markov_chain_mc(self, N, n=None):
        for group_step in range(N):
            if self.i_step % 100:
                 self.T *= 0.95
            self.i_step += 1
            print(f'step {n}, energy {self.energy()}, temperature {self.T}')
            for i, particle in self.particles.items():
                step = np.random.uniform(-self.step_size, self.step_size, size=(2,))
                pos = particle.vec()
                before_energy = self.energy()

                while not particle.update(pos + step): # give new position, and check if allowed, else make new step
                    step = np.random.uniform(-self.step_size, self.step_size, size=(2,))

                after_energy = self.energy()
                delta_energy = after_energy - before_energy

                if delta_energy > 0 and np.random.rand() > np.exp(-delta_energy/self.T):
                    particle.update(pos) # give old position
                    assert self.energy() == before_energy, 'Reset has failed'

    
    def forces(self):
        for i, particle in self.particles.items():
            force = particle.force(self.particles)
            print(i, force)
            # not finished

    def energy(self):
        total_energy = 0
        for i, particle in self.particles.items():
            total_energy += particle.energy(self.particles)
        return total_energy/self.n_particles 

    def plot(self):
        fig, axis = plt.subplots(1,1)

        # plot circle
        circle = np.linspace(0, 2*np.pi, 1000)
        circle_x = np.cos(circle)
        circle_Y = np.sin(circle)
        plt.plot(circle_x, circle_Y)

        # # plot particles
        # for i, particle in self.particles.items():
        #     x, y = particle.get_xy()
        #     axis.scatter(x, y, label=f'{i}')

        # plot particles
        points = []
        for i, particle in self.particles.items():
            points.append(particle.vec())

        points = np.array(points)
        axis.scatter(points[:, 0], points[:, 1], label=f'{i}')
        plt.show()
    
    def update(self, n):
        self.markov_chain_mc(1, n=n)
        points = []
        for i, particle in self.particles.items():
            points.append(particle.vec())

        points = np.array(points)
        self.scat.set_offsets(points)
        return self.scat

    def animate(self):
        fig, axis = plt.subplots(1,1)

        # plot circle
        circle = np.linspace(0, 2*np.pi, 1000)
        circle_x = np.cos(circle)
        circle_Y = np.sin(circle)
        plt.plot(circle_x, circle_Y)

        # plot particles
        points = []
        for i, particle in self.particles.items():
            points.append(particle.vec())

        points = np.array(points)
        self.scat = axis.scatter(points[:, 0], points[:, 1], label=f'{i}')
        
        ani = animation.FuncAnimation(fig=fig, func=self.update, frames=11, interval=30)
        plt.gca().set_aspect('equal')
        plt.show()




class particle():

    def __init__(self, i) -> None:
        self.id = i
        self.r = np.random.rand()
        self.theta = np.random.rand()*2*np.pi
        self.x = self.r*np.cos(self.theta)
        self.y = self.r*np.sin(self.theta)
    
    def get_xy(self):
        return self.x, self.y

    def vec(self):
        return np.array([self.x, self.y])

    def force(self, particles):
        # force of circle boundary (radius = 1)
        dist_to_circle = 1 - self.r
        #force_circle = 1/dist_to_circle**5 # 1/r^3 falls of quickly
        force_circle = -10*np.exp(-10*dist_to_circle**2) # falls of quickly

        pos = self.vec()
        norm = np.sqrt(pos.dot(pos))


        total_force = - pos/norm * force_circle # start with inward force from circle boundary
        # force of particles
        for i, particle in particles.items():
            if i != self.id:

                #dist_to_particle = np.sqrt((x_other - x_self)**2 + (y_other - y_self)**2)
                vec_to_particle = particle.vec() - pos 
                dist_to_particle = np.sqrt(vec_to_particle.dot(vec_to_particle))
                force_to_particle = vec_to_particle/dist_to_particle**3

                total_force += force_to_particle
        return total_force

    def energy(self, particles):
        pos = self.vec()
        energy = 0
        for i, particle in particles.items():
            if i != self.id and i > self.id: # only check i > id to avoid duplicates

                #dist_to_particle = np.sqrt((x_other - x_self)**2 + (y_other - y_self)**2)
                vec_to_particle = particle.vec() - pos 
                dist_to_particle = np.sqrt(vec_to_particle.dot(vec_to_particle))

                energy += 1/dist_to_particle
        return energy

    def update(self, new_pos):
        self.x = new_pos[0]
        self.y = new_pos[1]
        self.theta = np.arctan(self.y/self.x)
        self.r = np.sqrt(self.x**2 + self.y**2)
        if 1 - self.r <= 0:
            return False
        else: 
            return True

sim = sim(16)
#sim.markov_chain_mc(20)
sim.animate()