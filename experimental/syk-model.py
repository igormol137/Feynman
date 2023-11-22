import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BoltzmannMachineSYK:
    def __init__(self, num_units, initial_state=None):
        self.num_units = num_units
        self.weights = np.random.randn(num_units, num_units)
        self.biases = np.random.randn(num_units)
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = np.random.choice([1, -1], size=num_units)

    def energy(self):
        return -0.5 * np.dot(self.state, np.dot(self.weights, self.state)) - np.dot(self.biases, self.state)

    def probability(self, energy, temperature):
        return np.exp(-energy / temperature)

    def update_state(self, temperature):
        new_state = self.state.copy()
        for i in range(self.num_units):
            energy_change = 2 * self.state[i] * (np.dot(self.weights[i, :], self.state) + self.biases[i])
            if np.random.rand() < self.probability(energy_change, temperature):
                new_state[i] = -new_state[i]
        self.state = new_state

    def simulate(self, num_steps, temperature):
        for _ in range(num_steps):
            self.update_state(temperature)

def plot_configuration(final_state):
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(final_state)), final_state, color=['blue' if s == 1 else 'red' for s in final_state])
    plt.title('Final Configuration of the Boltzmann Machine')
    plt.xlabel('Unit Index')
    plt.ylabel('State')
    plt.show()

if __name__ == "__main__":
    # Load initial data from CSV file
    csv_file_path = "/Users/igormol/Desktop/Feynman/experimental/fermion-network.csv"
    initial_data = pd.read_csv(csv_file_path)
    initial_state = initial_data.values[0]

    num_units = len(initial_state)
    num_steps = 1000
    temperature = 1.0

    bm = BoltzmannMachineSYK(num_units, initial_state)
    bm.simulate(num_steps, temperature)

    print("Final state:", bm.state)
    print("Final energy:", bm.energy())

    # Plot the final configuration
    plot_configuration(bm.state)
