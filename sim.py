'''Code that will contain some comparison plots'''

import numpy as np
import time 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import pickle

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class sim():

    def __init__(self, n, schedule = 'default'):	
        self.schedule = schedule
        self.n_particles = n
        self.particles = {}
        self.ani = None
        for i in range(self.n_particles):
            self.particles[i] = particle(i)
        # start temp
        self.T = 1000
        self.T0 = 1000
        self.i_step = 1
        self.energy_list = [self.energy()]
        self.temperature_list = [self.T0]
        self.specific_heat_list = [0]
        self.step_size = 0.06 # 0.06
        self.step_size0 = 0.06 # 0.06

    
    def step(self, particle):
        force = particle.force(self.particles, self.i_step)
        force_norm = np.sqrt(force.dot(force))
        rand = np.random.uniform(-1, 1, size=(2,))
        rand_norm = np.sqrt(rand.dot(rand))
        step = (0.95 * rand/rand_norm + 0.05 * force/force_norm) * self.step_size
        # step = (0.1 * force/force_norm) * self.step_size
        # step = rand/rand_norm * self.step_size
        # step = 0.9*np.random.uniform(-self.step_size, self.step_size, size=(2,)) + 0.05*(force/force_norm * np.random.uniform(0, self.step_size))
        # step = (force/force_norm * np.random.uniform(0, self.step_size))
        return step

    def markov_chain_mc(self, N, n=None, schedule='default', alpha=0.95):
        start = time.time()
        for group_step in range(N):
            if group_step % 100 == 0 and N > 1:
                now = time.time()
                #print("current time elapsed: ", now - start)
                print('E', self.energy(), 'step', self.i_step, 'step size', self.step_size, 'temp', self.T, self.end_config())
                # self.plot()

            for i, particle in self.particles.items():
                if self.i_step % 200 == 0:
                    if schedule == 'linear':
                        self.T = np.max(self.T0 - alpha * self.i_step)
                        self.step_size = np.max(self.step_size0 - alpha * self.i_step, 0.001)

                    elif schedule == 'exponential':
                        self.T = self.T0 * (alpha ** self.i_step)
                        self.step_size = self.step_size0 * (alpha ** self.i_step)

                    elif schedule == 'logarithmic':
                        self.T = self.T0 / (np.log(1 + self.i_step))
                        self.step_size = max(self.step_size0 / (np.log(1 + self.i_step)), 0.001)

                        #self.step_size = C / (np.log(1 + self.i_step))
                    elif schedule == 'default':
                        self.T = self.T * 0.9
                        self.step_size = self.step_size * 0.99

                self.i_step += 1

                pos = particle.vec()
                # before_energy = self.energy()
                before_energy = self.energy_list[-1]

                #print(self.step_size, self.i_step, self.T, before_energy)

                # self.energy_list.append(before_energy)
                self.temperature_list.append(self.T)

                E = np.array(self.energy_list)
                specific_heat = (E.dot(E) / len(E) - np.mean(E) ** 2)
                self.specific_heat_list.append(specific_heat/10)

                step = self.step(particle)
                ntry = 0
                while not particle.update(pos + step) and ntry < 50:
                    ntry += 1
                        #new random* step
                    step = self.step(particle)


                after_energy = self.energy()
                delta_energy = after_energy - before_energy

                p = np.exp(-delta_energy / self.T, dtype='d')
                if delta_energy > 0 and np.random.rand() > p:
                        particle.update(pos)
                        self.energy_list.append(before_energy)
                else:
                        self.energy_list.append(after_energy)
                

            if len(self.energy_list) > 10 and all([np.abs(self.energy_list[-i] - self.energy_list[-i-1]) < 0.0001 for i in range(1, self.n_particles-1)]):
                if self.ani is not None:
                    print('E', self.energy(), 'step', self.i_step, 'step size', self.step_size, 'temp', self.T, self.end_config())
                    self.ani.pause()
                else:
                    return True

    def energy(self):
        total_energy = 0
        for i, particle in self.particles.items():
            total_energy += particle.energy(self.particles)
        return total_energy #/self.n_particles 

    def plot(self):
        fig, axis = plt.subplots(1,2)

        # axis[1].scatter( self.temperature_list, np.array(self.energy_list)/10)
        # axis[1].scatter(self.temperature_list, self.specific_heat_list)
        axis[1].plot(range(self.i_step-1), np.array(self.energy_list))
        axis[1].plot(range(self.i_step-1), self.specific_heat_list)
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
        self.sl[0].set_offsets(points)
        self.sl[2].set_data(self.temperature_list, self.energy_list)
        self.sl[1].set_data(self.temperature_list, self.specific_heat_list)

        self.ax2.set_ylim(min(self.specific_heat_list), max(self.specific_heat_list))
        self.ax2.set_xlim(1e-3, 1e3)
        self.ax2.set_xscale('log')

        # return [self.scat]
        return self.sl

    def animate(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        # fig, ax1 = plt.subplots(1,1)
        self.ax2 = ax2
        self.ax2.set_xscale('log')

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
        scat = ax1.scatter(points[:, 0], points[:, 1], label=f'{i}')
        line2, = ax2.plot(self.temperature_list, self.energy_list)
        line, = ax2.plot(self.temperature_list, self.specific_heat_list)
        self.sl = [scat, line, line2]

        
        self.ani = animation.FuncAnimation(fig=fig, func=self.update, frames=10, blit=True, interval=30)
        ax1.set_aspect('equal')
        plt.show()

    def end_config(self):
        radii = []
        middle = 0
        for i, particle in self.particles.items():
            r = particle.r
            radii.append(r)
            if r < 0.8:
                middle += 1
        return middle


class particle():

    def __init__(self, i) -> None:
        self.id = i
        self.r = np.random.rand()
        self.theta = np.random.rand()*2*np.pi
        self.x = self.r*np.cos(self.theta)
        self.y = self.r*np.sin(self.theta)
        self.last_i = -1
        self.last_force = 0
    
    def get_xy(self):
        return self.x, self.y

    def vec(self):
        return np.array([self.x, self.y])

    def force(self, particles, i):
        # if force is needed for same particle at same step. i.e step was outside bc. reuse force from last calc
        if i == self.last_i:
            return self.last_force

        pos = self.vec()

        total_force = 0
        # force of particles
        for i, particle in particles.items():
            if i != self.id:

                #dist_to_particle = np.sqrt((x_other - x_self)**2 + (y_other - y_self)**2)
                vec_to_particle = pos - particle.vec()
                dist_to_particle = np.sqrt(vec_to_particle.dot(vec_to_particle))
                force_to_particle = vec_to_particle/dist_to_particle**3

                total_force += force_to_particle

        self.last_i = i
        self.last_force = total_force

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


def calc_mean(n_sim, N, stop_N, mid, schedule='logarithmc'):
    """Runs the simulation N times to calculate and plot mean and stdev among 
    multiple runs with the same parameters.
    """
    
    df = pd.read_csv('results.csv')

    #Store all simulations in a list
    sims = []

    #Number of simulations with accepted outcome
    simulations = 0
    while simulations != n_sim:
        s = sim(N, schedule)
        s.markov_chain_mc(stop_N)
        end_config = s.end_config()
        #Only accept simulations with correct number of center particles
        if end_config == mid:
            simulations += 1
            sims.append(s)
            print("accepted mid: ", end_config)
            df = pd.concat([df, pd.DataFrame.from_dict(data={'E': [s.energy_list[-1]], 'N':[s.n_particles], 'Middle':[s.end_config()]})], ignore_index=True)
        else:
            print("not accepted mid: ", end_config)

    df.to_csv('results.csv', index=False)

    #Minimum length (for plotting, mean and stdev)
    min_length = min(len(s.temperature_list) for s in sims)

    energy_end = []
    energy_mean = []
    energy_stdev = []

    #Truncate all lists to minimum length for plotting
    for s in sims:
        energy_end.append(s.energy_list[-1])
        s.energy_list = s.energy_list[:min_length]
        s.specific_heat_list = s.specific_heat_list[:min_length]

    #Calculate mean and standard deviation per simulation
    energy_array = np.array([s.energy_list for s in sims])
    energy_mean = np.mean(energy_array, axis=0)
    energy_std = np.std(energy_array, axis=0)

    sh_array = np.array([s.specific_heat_list for s in sims])
    sh_mean = np.mean(sh_array, axis=0)
    sh_std = np.std(sh_array, axis=0)

    iterations = list(range(len(energy_mean)))

    fontsize = 17
    #Plot mean and stdev.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'hspace': 0})

    #Plot energy vs Temp.
    ax1.plot(iterations, energy_mean, label='Energy Mean', color='tab:blue')
    ax1.fill_between(iterations, energy_mean - energy_std, 
                     energy_mean + energy_std, color='tab:blue', 
                     alpha=0.2, label='Energy Std')
    ax1.set_ylabel('Energy', fontsize=fontsize)
    ax1.legend(fontsize=fontsize - 2)
    ax1.grid(True)
    ax1.tick_params(labelbottom=False, labelsize=fontsize - 2)
    ax2.tick_params(labelsize=fontsize - 2)
    #Plot specific heat vs Temp.
    ax2.plot(iterations, sh_mean, label='Specific Heat Mean', color='tab:green')
    ax2.fill_between(iterations, sh_mean - sh_std, sh_mean + sh_std, 
                     color='tab:green', alpha=0.2, label='Specific Heat Std')
    ax2.set_xlabel('Iterations', fontsize=fontsize)
    ax2.set_ylabel('Specific Heat', fontsize=fontsize)
    ax2.legend(fontsize=fontsize - 2)
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'Figures/N_{N}_nsim{n_sim}_{schedule}.pdf', bbox_inches='tight', format='pdf')
    plt.show()


n_sim = 5
N = 12
stop_N = 5000

#Signifies expected number of particles not on the ring
mid = 1

calc_mean(n_sim, 12, stop_N, 1, schedule='logarithmic')
calc_mean(n_sim, 16, stop_N, 2, schedule='logarithmic')
calc_mean(n_sim, 17, stop_N, 3, schedule='logarithmic')

# 16: 3-circle, 116.57

# sim = sim(11, schedule= 'linear')
# sim = sim(11, schedule= 'exponential')
# sim = sim(16, schedule= 'logarithmic')

#sim = sim(39, schedule= 'logarithmic')
#sim.animate()

'''# sim.markov_chain_mc(5000)
print('------------- \n end energy: ', sim.energy(), 'step: ', sim.i_step, '\n ------------------')
plt.plot( sim.temperature_list, np.array(sim.energy_list), label='Total system energy')
plt.plot(sim.temperature_list, sim.specific_heat_list, label='Specific Heat')
# plt.plot(range(sim.i_step-1), sim.specific_heat_list, label='Specific Heat 2')
plt.xscale('log')
# plt.yscale('log')
plt.show()
# sim.plot()

# print(df)
# sim.plot()'''

df = pd.read_csv('results.csv')

#print(df[df['N'] == 39])
# simm = sim(21, schedule= 'logarithmic')
# simm.markov_chain_mc(5000)
# minimum_length = len(simm.temperature_list)
'''
for N in range(12, 41):
    for _ in range(8):
        simm = sim(N, schedule= 'logarithmic')
        simm.markov_chain_mc(5000)
        length = len(simm.temperature_list)
    # if len(simm.temperature_list) < :
            #min_length = 
        #print("Length temp list: ", len(simm.temperature_list))
        df_simm = pd.DataFrame.from_dict(data={'E': [simm.energy_list[-1]], 'N':[simm.n_particles], 'Middle':[simm.end_config()]})
        df = pd.concat([df, df_simm], ignore_index=True)
        print(df_simm)
        print("t at 1000 ", simm.temperature_list[1000])
        #energy = np.array(simm.energy_list)
        #energy_mean = np.mean(energy)
        #energy_stdev = np.std(energy)
        # clean sim for rerun
        print(df[df['N'] == N]['E'])
        if sum(df['N'] == N) == 0 or all(df[df['N'] == N]['E'] >= simm.energy_list[-1]):
            with open(f'sims/sim_obj_{N}.obj', 'wb') as f:
                print('save obj')
                pickle.dump(simm, f)



        #print(sim.energy_list)
        df.to_csv('results.csv', index=False)
        '''