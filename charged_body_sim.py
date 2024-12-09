import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class sim():

    def __init__(self, n):
        self.n_particles = n
        self.particles = {}
        for i in range(self.n_particles):
            self.particles[i] = particle(i)

        # start temp
        self.T = 1000
        self.T0 = 1000
        self.i_step = 1
        self.energy_list = []
        self.temperature_list = []
        self.specific_heat_list = []
        self.step_size = 0.06
        self.step_size0 = 0.06
    
    def step(self, particle):
        # step_size = 2/np.log(2*self.i_step)
        # step dirction is 50/50 random and by force
        # isn't actually 50/50 bc. random vec norm can be bigger than stepsize
        # step = 0.9*np.random.uniform(-self.step_size, self.step_size, size=(2,)) - 0.1*(force/force_norm * self.step_size)
        force = particle.force(self.particles)
        force_norm = np.sqrt(force.dot(force))
        #step = force/force_norm * self.i_step
        step = 0.9*np.random.uniform(-self.step_size, self.step_size, size=(2,)) - 0.1*(force/force_norm * self.step_size)
        # step = 1*np.random.uniform(-self.step_size, self.step_size, size=(2,)) - 0.1*(force/force_norm * self.step_size)
        #step = np.random.uniform(-self.step_size, self.step_size, size=(2,))
        return step


    def markov_chain_mc(self, N, n=None):
        for group_step in range(N):

            ''' #Less simple annealing
            if self.i_step % 50 == 0:
                self.T = self.T_initial / (1 + 0.2 * self.i_step)
                self.step_size = self.step_size_initial / (np.log(1 + self.i_step))'''

            if group_step % 100 == 0 and N > 1:
                # show info and plot of run during sim every ~1min
                print('E', self.energy(), 'step', self.i_step, 'step size', self.step_size)
                self.plot()

            for i, particle in self.particles.items():
                
                # Simple annealing
                # annealing of temperature
                if self.i_step % 100 == 0:
                    self.T *= 0.9
                    self.step_size *= 0.99

                self.i_step += 1

                # step size decreases as amount of steps grows.
                # 0.3 naar 0.001
                # 1 to 1800
                # if self.step_size <= 0.0003:
                #     np.log(self.step_size)
                #     self.step_size -= 0.0000001
                # else:
                #     self.step_size -= 0.0000100  
                # if self.step_size < 0:
                #     return False

                pos = particle.vec()
                before_energy = self.energy()

                # run info 
                print(self.step_size, self.i_step, self.T, before_energy)

                self.energy_list.append(before_energy)
                self.temperature_list.append(self.T)

                E = np.array(self.energy_list)
                specific_heat = (E.dot(E)/len(E) - np.mean(E)**2 ) # / self.T / self.T

                self.specific_heat_list.append(specific_heat)
                # print(f'step {self.i_step}, energy {self.energy()}, temperature {self.T}, specific_heat {specific_heat}')


                step = self.step(particle)
                while not particle.update(pos + step): # give new position, and check if allowed, else make new step
                    step = self.step(particle)

                # energy after step
                after_energy = self.energy()

                # difference in energy
                delta_energy = after_energy - before_energy

                if delta_energy > 0:
                    p = np.exp(-delta_energy/self.T, dtype= 'd')
                    # assert p != 0
                    if np.random.rand() > p: # > it is the chance of rejection !! set it back if true
                        particle.update(pos) # give old position
                        # assert self.energy() == before_energy, 'Reset has failed'

            if self.specific_heat_list[-1] < 0.1:
                return True

    def energy(self):
        total_energy = 0
        for i, particle in self.particles.items():
            total_energy += particle.energy(self.particles)
        return total_energy #/self.n_particles 

    def plot(self):
        fig, axis = plt.subplots(1,2)

        axis[1].plot( self.temperature_list, np.array(self.energy_list)/10)
        axis[1].plot(self.temperature_list, self.specific_heat_list)
        axis[1].set_xscale('log')

        # plot circle
        circle = np.linspace(0, 2*np.pi, 1000)
        circle_x = np.cos(circle)
        circle_Y = np.sin(circle)
        axis[0].plot(circle_x, circle_Y)

        # # plot particles
        # for i, particle in self.particles.items():
        #     x, y = particle.get_xy()
        #     axis.scatter(x, y, label=f'{i}')

        # plot particles
        points = []
        for i, particle in self.particles.items():
            points.append(particle.vec())

        points = np.array(points)
        axis[0].scatter(points[:, 0], points[:, 1], label=f'{i}')
        axis[0].axis('equal')
        plt.show()
    
    def update(self, n):
        self.markov_chain_mc(1, n=n)
        points = []
        for i, particle in self.particles.items():
            points.append(particle.vec())

        points = np.array(points)
        # self.sl[0].set_offsets(points)
        # self.sl[1].set_data(list(range(len(self.energy_list))), self.energy_list)
        self.scat.set_offsets(points)


        return [self.scat]

    def animate(self):
        # fig, (ax1, ax2) = plt.subplots(1,2)
        fig, ax1 = plt.subplots(1,1)

        # plot circle
        circle = np.linspace(0, 2*np.pi, 1000)
        circle_x = np.cos(circle)
        circle_Y = np.sin(circle)
        ax1.plot(circle_x, circle_Y)

        # plot particles
        points = []
        for i, particle in self.particles.items():
            points.append(particle.vec())

        points = np.array(points)
        self.scat = ax1.scatter(points[:, 0], points[:, 1], label=f'{i}')
        # self.line = ax2.plot([0,1] , [0,1])[0]
        # self.sl = [self.scat, self.line]
        
        self.ani = animation.FuncAnimation(fig=fig, func=self.update, frames=10, blit=True, interval=30)
        ax1.set_aspect('equal')
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
        pos = self.vec()

        total_force = 0
        # force of particles
        for i, particle in particles.items():
            if i != self.id:

                #dist_to_particle = np.sqrt((x_other - x_self)**2 + (y_other - y_self)**2)
                vec_to_particle = pos - particle.vec()
                dist_to_particle = np.sqrt(vec_to_particle.dot(vec_to_particle))
                force_to_particle = vec_to_particle/dist_to_particle**3

                total_force -= force_to_particle

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
        x = new_pos[0]
        y = new_pos[1]
        r = np.sqrt(x**2 + y**2)
        if 1 - r <= 0:
            return False
        self.x = x
        self.y = y
        self.theta = np.arctan(self.y/self.x)
        self.r = r
        return True


# 16: 3-circle, 116.57

sim = sim(11)
sim.animate()

sim.markov_chain_mc(5000)
"""
print('------------- \n end energy: ', sim.energy(), 'step: ', sim.i_step, '\n ------------------')
plt.plot( sim.temperature_list, np.array(sim.energy_list)/10)
plt.plot(sim.temperature_list, sim.specific_heat_list)
plt.xscale('log')
# plt.yscale('log')
plt.show()
sim.plot()"""