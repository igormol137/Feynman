import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_energy(configuration, coupling_matrix):
    return -0.5 * np.sum(configuration * np.dot(coupling_matrix, configuration))

def metropolis_algorithm(configuration, coupling_matrix, temperature):
    new_configuration = configuration.copy()
    random_site = np.random.randint(len(configuration))
    new_configuration[random_site] *= -1  # Flip the spin at the randomly chosen site

    delta_energy = 2 * configuration[random_site] * np.dot(coupling_matrix[random_site, :], configuration)

    if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
        return new_configuration
    else:
        return configuration

def simulate_spin_glass_system(initial_configuration, coupling_matrix, temperature, num_steps):
    current_configuration = initial_configuration.copy()
    configurations = [current_configuration]

    for _ in range(num_steps):
        current_configuration = metropolis_algorithm(current_configuration, coupling_matrix, temperature)
        configurations.append(current_configuration)

    return np.array(configurations)

def plot_configuration(configuration):
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(configuration)), configuration, color=['blue' if s == 1 else 'red' for s in configuration])
    plt.title('Spin Glass System Configuration')
    plt.xlabel('Spin Index')
    plt.ylabel('Spin State')
    plt.show()

if __name__ == "__main__":
    # Load initial data from CSV file
    csv_file_path = "/Users/igormol/Desktop/Feynman/experimental/fermion-network.csv"
    initial_data = pd.read_csv(csv_file_path)
    initial_configuration = initial_data.values[0]

    num_spins = len(initial_configuration)
    num_steps = 1000
    temperature = 1.0

    # Generate a random coupling matrix for the spin glass system
    coupling_matrix = np.random.randn(num_spins, num_spins)

    # Symmetrize the coupling matrix
    coupling_matrix = 0.5 * (coupling_matrix + coupling_matrix.T)

    # Simulate the spin glass system using the Metropolis algorithm
    configurations = simulate_spin_glass_system(initial_configuration, coupling_matrix, temperature, num_steps)

    # Plot the final configuration
    plot_configuration(configurations[-1])
