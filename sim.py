"""
This module simulates the behavior of particles under various schedules 
using a Markov Chain Monte Carlo (MCMC) method. It includes 
visualization, energy, and variance calculations to study particle 
interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import pickle

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class sim:
    """
    Class to simulate the behavior of particles using a Markov Chain 
    Monte Carlo (MCMC) method.

    Methods:
    step(particle):
        Calculates the step vector for a given particle based on random 
        motion and inter-particle forces.
    markov_chain_mc(N, n=None, schedule='default', alpha=0.95, 
                    markov_chain_length=200):
        Performs the MCMC simulation for a specified number of steps and 
        updates the system.
    energy():
        Calculates and returns the total energy of the system.
    plot():
        Visualizes the system's energy, specific heat trends, and 
        particle configuration.
    update(n):
        Updates the visualization for a single frame of the animation.
    animate():
        Creates and displays an animated visualization of the simulation.
    end_config():
        Returns the number of particles located near the center of the 
        circular boundary.
    """


    def __init__(self, n, schedule='default'):
        """
        Initializes the simulation with the given number of particles 
        and schedule.

        Parameters:
            n (int): Number of particles.
            schedule (str): Schedule type for temperature adjustment.
        """
        self.schedule = schedule
        self.n_particles = n
        self.particles = {}
        self.ani = None

        for i in range(self.n_particles):
            self.particles[i] = particle(i)

        self.T = 1000 #Start temp
        self.T0 = 1000
        self.i_step = 1
        self.energy_list = [self.energy()]
        self.temperature_list = [self.T0]
        self.specific_heat_list = [0]
        self.step_size = 0.06 # 0.06
        self.step_size0 = 0.06 # 0.06

    
    def step(self, particle):
        """
        Calculate the step vector for a given particle.

        Parameters:
            particle (particle): A particle object.

        Returns the step vector as an array.
        """
        force = particle.force(self.particles, self.i_step)
        force_norm = np.sqrt(force.dot(force))

        rand = np.random.uniform(-1, 1, size=(2,))
        rand_norm = np.sqrt(rand.dot(rand))

        step = self.step_size * (0.95 * rand/rand_norm 
                                 + 0.05 * force/force_norm)

        return step


    def markov_chain_mc(self, N, n=None, schedule='default', alpha=0.95, 
                        markov_chain_length=200):
        """
        Performs the Markov Chain Monte Carlo (MCMC) simulation.

        Parameters:
            N (int): Number of simulation steps.
            n (int, optional): Additional parameter.
            schedule (str): Schedule type for temperature adjustment.
            alpha (float): Parameter controlling the cooling schedule.
            markov_chain_length (int): Length of the Markov chain for 
            updates.
        """
        for group_step in range(N):
            if group_step % 150 == 0 and N > 1:

                print('E', self.energy(), 'step', self.i_step, 'step size', 
                      self.step_size, 'temp', self.T, self.end_config())

            for i, particle in self.particles.items():
                if self.i_step % markov_chain_length == 0:
                    if schedule == 'linear':
                        self.T = np.max(self.T0 - alpha*self.i_step)
                        self.step_size = np.max(self.step_size0 - alpha 
                                                * self.i_step, 0.001)

                    elif schedule == 'exponential':
                        self.T = self.T0 * (alpha**self.i_step)
                        self.step_size = self.step_size0 * (alpha**self.i_step)

                    elif schedule == 'logarithmic':
                        self.T = self.T0 / (np.log(1 + self.i_step))
                        self.step_size = max(self.step_size0 
                                            / (np.log(1 + self.i_step)), 0.001)

                    elif schedule == 'default':
                        self.T = self.T * 0.9
                        self.step_size = self.step_size * 0.99

                self.i_step += 1

                pos = particle.vec()

                before_energy = self.energy_list[-1]
                self.temperature_list.append(self.T)
                E = np.array(self.energy_list)

                specific_heat = (E.dot(E) / len(E) - np.mean(E) ** 2)
                self.specific_heat_list.append(specific_heat)

                step = self.step(particle)

                ntry = 0
                while not particle.update(pos + step) and ntry < 50:
                    ntry += 1
                    step = self.step(particle)

                after_energy = self.energy()
                delta_energy = after_energy - before_energy

                p = np.exp(-delta_energy / self.T, dtype='d')
                if delta_energy > 0 and np.random.rand() > p:
                        particle.update(pos)
                        self.energy_list.append(before_energy)

                else:
                    self.energy_list.append(after_energy)

            if len(self.energy_list) > 10 and all(
                    [np.abs(self.energy_list[-i] - self.energy_list[-i-1]) 
                    < 0.0001 for i in range(1, self.n_particles-1)]):

                if self.ani is not None:
                    print('E', self.energy(), 'step', self.i_step, 'step size',
                          self.step_size, 'temp', self.T, self.end_config())
                          
                    self.ani.pause()

                else:
                    return True


    def energy(self):
        """Calculates and returns the total energy of the system."""
        total_energy = 0
        for i, particle in self.particles.items():
            total_energy += particle.energy(self.particles)

        return total_energy 


    def plot(self):
        """
        Plot the simulation data including energy and specific heat 
        trends over the simulation steps, as well as the spatial 
        configuration of particles within a circular boundary.

        This method creates a two-panel plot:
        1. A scatter plot showing the positions of particles overlaid on
        a unit circle.
        2. A line plot displaying energy and specific heat values versus 
        iteration steps on a logarithmic x-axis.
        """
        fig, axis = plt.subplots(1,2)

        axis[1].plot(range(self.i_step), np.array(self.energy_list))
        axis[1].plot(range(self.i_step), self.specific_heat_list)
        axis[1].set_xscale('log')

        #Plot circle
        circle = np.linspace(0, 2*np.pi, 1000)
        circle_x = np.cos(circle)
        circle_Y = np.sin(circle)
        axis[0].plot(circle_x, circle_Y)

        # plot particles
        points = []
        for i, particle in self.particles.items():
            points.append(particle.vec())

        points = np.array(points)
        axis[0].scatter(points[:, 0], points[:, 1], label=f'{i}')
        axis[0].axis('equal')
        plt.show()
    

    def update(self, n):
        """
        Update the simulation visualization for a single animation frame.

        This method performs a single step of the simulation using the 
        Markov Chain Monte Carlo (MCMC) method, updates the positions of 
        particles, and refreshes the plot elements to reflect the new 
        state.

        Parameters:
        n (int): The current frame number for the animation (not 
            explicitly used within the method but required by the
            animation framework).

        Returns a list of plot elements (`self.sl`) that have been 
        updated. This is required for Matplotlib's animation framework.
        """
        self.markov_chain_mc(1, n=n)
        points = []
        for i, particle in self.particles.items():
            points.append(particle.vec())

        points = np.array(points)

        self.sl[0].set_offsets(points)
        self.sl[2].set_data(self.temperature_list, self.energy_list)
        self.sl[1].set_data(self.temperature_list, self.specific_heat_list)

        self.ax2.set_ylim(min(self.specific_heat_list), 
                          max(self.specific_heat_list))
        self.ax2.set_xlim(1e-3, 1e3)
        self.ax2.set_xscale('log')

        return self.sl


    def animate(self):
        """
        Creates and displays an animated visualization of the simulation.

        Visualization Details:
        - Left plot: Shows a circular boundary and the current positions 
        of particles.
        - Right plot: Displays the system's energy and specific heat as 
        functions of time (logarithmic scale).
        """
        fig, (ax1, ax2) = plt.subplots(1,2)
        self.ax2 = ax2
        self.ax2.set_xscale('log')

        #Plot circle
        circle = np.linspace(0, 2*np.pi, 1000)
        circle_x = np.cos(circle)
        circle_Y = np.sin(circle)
        ax1.plot(circle_x, circle_Y)

        #Plot particles
        points = []
        for i, particle in self.particles.items():
            points.append(particle.vec())

        points = np.array(points)
        scat = ax1.scatter(points[:, 0], points[:, 1], label=f'{i}')
        line2, = ax2.plot(self.temperature_list, self.energy_list)
        line, = ax2.plot(self.temperature_list, self.specific_heat_list)

        self.sl = [scat, line, line2]
        self.ani = animation.FuncAnimation(fig=fig, func=self.update, 
                                           frames=10, blit=True, interval=30)
        ax1.set_aspect('equal')
        plt.show()


    def end_config(self):
        """Returns the number of particles present in the middle of 
        the circle."""
        
        radii = []
        middle = 0

        for i, particle in self.particles.items():
            r = particle.r
            radii.append(r)

            if r < 0.8:
                middle += 1

        return middle


class particle():
    """
    A class to represent a particle in a system interacting with other 
    particles.

    Methods:
    get_xy():
        Returns the current (x, y) coordinates.
    vec():
        Returns the particle's position as a NumPy array.
    force(particles, i):
        Computes and returns the net force on the particle due to other 
        particles.
    energy(particles):
        Computes and returns the potential energy of the particle with 
        others.
    update(new_pos):
        Updates the particle's position if within valid bounds.
    """

    def __init__(self, i) -> None:
        """
        Initialize a particle with random position inside a unit circle.
        
        Parameters:
        i (int): Unique identifier for the particle.
        """
        self.id = i
        self.r = np.random.rand()
        self.theta = np.random.rand()*2*np.pi
        self.x = self.r*np.cos(self.theta)
        self.y = self.r*np.sin(self.theta)
        self.last_i = -1
        self.last_force = 0
    

    def get_xy(self):
        """Returns the (x, y) coordinates of the particle as a tuple."""
        return self.x, self.y


    def vec(self):
        """Returns the particle's position as a NumPy array."""
        return np.array([self.x, self.y])


    def force(self, particles, i):
        """
        Computes the net force acting on the particle due to 
        interactions with others.

        Parameters:
        particles (dict): A dictionary of other particles in the system.
        i (int): The current simulation step index.

        Returns a 2D array representing the net force vector.
        """
        #Reuse previous force if step was outside
        if i == self.last_i:
            return self.last_force

        pos = self.vec()
        total_force = 0

        #Force of particles
        for i, particle in particles.items():
            if i != self.id:

                vec_to_particle = pos - particle.vec()
                dist_to_particle = np.sqrt(vec_to_particle.dot(vec_to_particle))
                force_to_particle = vec_to_particle/dist_to_particle**3

                total_force += force_to_particle

        self.last_i = i
        self.last_force = total_force

        return total_force


    def energy(self, particles):
        """
        Computes the potential energy of the particle due to interactions
        with others.

        Parameters:
        particles (dict): A dictionary of other particles in the system.

        Returns a float of the total potential energy.
        """
        pos = self.vec()
        energy = 0

        for i, particle in particles.items():
            #Only check i > id to avoid duplicates
            if i != self.id and i > self.id: 
                vec_to_particle = particle.vec() - pos 
                dist_to_particle = np.sqrt(vec_to_particle.dot(vec_to_particle))

                energy += 1/dist_to_particle

        return energy


    def update(self, new_pos):
        """
        Updates the particle's position if the new position is valid.

        Parameters:
        new_pos (tuple): A tuple (x, y) representing the new position.

        Returns a bool that is true if the update is successful, False 
        otherwise.
        """
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
    """
    Calculates and visualizes the mean and standard deviation of energy 
    and variance across multiple simulations, only accepting simulations 
    that converge to the specified number of central particles.

    Parameters:
    n_sim (int): Number of accepted simulations to perform.
    N (int): Number of particles in each simulation.
    stop_N (int): Number of steps to run for each simulation.
    mid (int): Desired number of particles near the center to consider a 
        simulation as accepted.
    schedule (string, optional): Cooling schedule type for temperature 
        adjustment. Default is 'logarithmc'.

    Returns a list containing the lengths of the energy lists for all 
    accepted simulations and the total number of simulations performed, 
    including rejected ones.

    Visualization:
    - Plots the mean and standard deviation of energy and variance 
      across accepted simulations.
    - Saves the plot as a PDF in the `Figures` directory, with a 
      filename reflecting the parameters.
    """
    df = pd.read_csv('results.csv')

    #Store all simulations in a list
    sims = []

    #Number of simulations with accepted outcome
    simulations = 0
    sim_count_total = 0
    while simulations != n_sim:
        s = sim(N, schedule)
        s.markov_chain_mc(stop_N)
        end_config = s.end_config()
        sim_count_total += 1
        
        #Only accept simulations with correct number of center particles
        if end_config == mid:
            simulations += 1
            sims.append(s)
            df = pd.concat([df, pd.DataFrame.from_dict(data={
                'E': [s.energy_list[-1]], 'N':[s.n_particles],
                'Middle':[s.end_config()]})], ignore_index=True)


    df.to_csv('results.csv', index=False)

    #Minimum length (for plotting, mean and stdev)
    min_length = min(len(s.temperature_list) for s in sims)

    energy_end = []
    energy_mean = []
    energy_stdev = []

    #Steps needed for convergence
    lengths = []

    #Truncate all lists to minimum length for plotting
    for s in sims:
        energy_end.append(s.energy_list[-1])
        lengths.append(len(s.energy_list))
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), 
                                   gridspec_kw={'hspace': 0})

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
    ax2.plot(iterations, sh_mean, label='Variance Mean', color='tab:green')
    ax2.fill_between(iterations, sh_mean - sh_std, sh_mean + sh_std, 
                     color='tab:green', alpha=0.2, label='Variance Std')
    ax2.set_xlabel('Iterations', fontsize=fontsize)
    ax2.set_ylabel('Variance', fontsize=fontsize)
    ax2.legend(fontsize=fontsize - 2)
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'Figures/N_{N}_nsim{n_sim}_{schedule}.pdf', 
                bbox_inches='tight', format='pdf')
    plt.show()

    return lengths, sim_count_total


#Parameters for run
n_sim = 2 #Number of accepted simulations
N = 12 #Number of particles
stop_N = 5000 #Maximum number of steps in a group

#Signifies expected number of particles not on the ring
mid = 1

# calc_mean(n_sim, 12, stop_N, 1, schedule='logarithmic')
# calc_mean(n_sim, 16, stop_N, 2, schedule='logarithmic')
# calc_mean(n_sim, 17, stop_N, 3, schedule='logarithmic')
lengths, total = calc_mean(n_sim, N, stop_N, mid, schedule='logarithmic')

print(np.mean(lengths), np.std(lengths), total, n_sim/total)


