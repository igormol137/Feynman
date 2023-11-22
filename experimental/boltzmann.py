import numpy as np
import pandas as pd

class BoltzmannMachine:
    def __init__(self, weights):
        self.weights = weights
        self.num_neurons = len(weights)

    def update_neurons(self, states, temperature):
        for i in range(self.num_neurons):
            activation = np.dot(self.weights[i], states)
            prob = 1 / (1 + np.exp(-2 * activation / temperature))
            states[i] = 1 if np.random.rand() < prob else -1

    def simulate(self, temperature, num_steps):
        states = np.random.choice([-1, 1], size=self.num_neurons)
        results = []

        for step in range(num_steps):
            self.update_neurons(states, temperature)
            mean_energy = -0.5 * np.dot(states, np.dot(self.weights, states))
            results.append([step + 1, mean_energy])

        return pd.DataFrame(results, columns=['Step', 'Mean Energy'])

def initialize_weights_from_file(file_path):
    return pd.read_csv(file_path, header=None).values

# Example usage
file_path = '/Users/igormol/Desktop/Feynman/experimental/spin-glass.csv'
temperature = 2.0
num_steps = 10000

initial_weights = initialize_weights_from_file(file_path)
boltzmann_machine = BoltzmannMachine(initial_weights)
simulation_results = boltzmann_machine.simulate(temperature, num_steps)

# Print results in a table
print(simulation_results)
