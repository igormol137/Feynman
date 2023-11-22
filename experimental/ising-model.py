import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, spin_lattice, temperature):
        self.spin_lattice = spin_lattice
        self.temperature = temperature

    def calculate_energy(self, i, j):
        size = self.spin_lattice.shape[0]
        spin_ij = self.spin_lattice[i, j]
        neighbors_sum = (
            self.spin_lattice[(i - 1) % size, j] +
            self.spin_lattice[(i + 1) % size, j] +
            self.spin_lattice[i, (j - 1) % size] +
            self.spin_lattice[i, (j + 1) % size]
        )
        return -spin_ij * neighbors_sum

    def metropolis_step(self):
        size = self.spin_lattice.shape[0]
        i, j = np.random.randint(0, size, size=2)
        current_energy = self.calculate_energy(i, j)

        # Flip the spin and calculate the new energy
        self.spin_lattice[i, j] *= -1
        new_energy = self.calculate_energy(i, j)

        # Metropolis acceptance criterion
        delta_energy = new_energy - current_energy
        if delta_energy > 0 and np.random.rand() > np.exp(-delta_energy / self.temperature):
            # Revert the spin flip if not accepted
            self.spin_lattice[i, j] *= -1

    def simulate(self, num_steps):
        for _ in range(num_steps):
            self.metropolis_step()

    def plot_spin_lattice(self):
        plt.imshow(self.spin_lattice, cmap='binary', interpolation='nearest')
        plt.title('Ising Model Spin Configuration')
        plt.show()

def initialize_spin_lattice_from_file(file_path):
    return pd.read_csv(file_path, header=None).values

# Example usage
file_path = '/Users/igormol/Desktop/Feynman/experimental/spin-glass.csv'
temperature = 2.0
num_steps = 10000

initial_spin_lattice = initialize_spin_lattice_from_file(file_path)
ising_model = IsingModel(initial_spin_lattice, temperature)
ising_model.simulate(num_steps)
ising_model.plot_spin_lattice()
