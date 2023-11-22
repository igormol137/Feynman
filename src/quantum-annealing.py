# Simulated Annealing Algorithm to Combinatorial Optimization
#
# Igor Mol <igor.mol@makes.ai>
#
# The following program implements a solution to the Traveling Salesman Problem 
# (TSP) using a simulated annealing algorithm. The TSP is a classic optimization 
# problem where a salesman aims to find the shortest possible route that visits 
# a set of cities and returns to the starting city. The code begins by defining 
# a TravelingSalesmanProblem class, initializing it with city coordinates from a
# CSV file, and calculating pairwise distances between cities. The implemented 
# simulated annealing algorithm gradually explores solution space, allowing for 
# occasional uphill moves to escape local optima. The algorithm maintains a 
# current route and iteratively generates candidate routes by randomly swapping 
# two cities. The acceptance or rejection of a candidate route is determined by 
# the Metropolis criterion, considering the change in total distance and a tem-
# perature parameter. The process continues for a specified number of iterations
#  gradually reducing the temperature, resulting in an approximate solution to 
# the TSP.
#     The simulated_annealing method encapsulates the core of the algorithm, 
# utilizing a temperature parameter to control the acceptance of suboptimal mo-
# ves. The method intelligently explores the solution space, updating the current
# route and best route whenever a shorter distance is encountered. The overall 
# goal is to find a global minimum in the total distance, representing an opti-
# mal TSP route. The algorithm's efficiency is attributed to its ability to ex-
# plore a diverse range of solutions and escape local minima, making it a power-
# ful optimization technique.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define a class for the Traveling Salesman Problem
class TravelingSalesmanProblem:
    def __init__(self, csv_filepath):
        # Initialize the class with city data from a CSV file
        self.city_data = pd.read_csv(csv_filepath)
        # Extract city coordinates from the data
        self.city_coordinates = self.city_data[['X', 'Y']].values
        # Calculate pairwise distances between cities using NumPy
        self.distances = np.linalg.norm(
            self.city_coordinates[:, np.newaxis, :] - self.city_coordinates[np.newaxis, :, :],
            axis=-1
        )

    def total_distance(self, route):
        # Calculate the total distance of a given TSP route using pairwise distances
        return np.sum(self.distances[route[:-1], route[1:]]) + self.distances[route[-1], route[0]]

    def generate_random_route(self):
        # Generate a random TSP route by permuting the indices of cities
        return np.random.permutation(np.arange(len(self.city_coordinates)))

    def simulated_annealing(self, initial_temperature=1000, cooling_rate=0.995, num_iterations=10000):
        # Simulated Annealing algorithm for solving the TSP
        num_cities = len(self.city_coordinates)
        current_route = self.generate_random_route()
        current_distance = self.total_distance(current_route)
        best_route = np.copy(current_route)
        best_distance = current_distance
        temperature = initial_temperature

        for iteration in range(num_iterations):
            # Generate a new candidate route by swapping two cities
            candidate_route = np.copy(current_route)
            i, j = np.random.choice(num_cities, size=2, replace=False)
            candidate_route[i], candidate_route[j] = candidate_route[j], candidate_route[i]

            # Calculate the new distance of the candidate route
            candidate_distance = self.total_distance(candidate_route)

            # Accept or reject the candidate based on the Metropolis criterion
            if candidate_distance < current_distance or np.random.rand() < np.exp((current_distance - candidate_distance) / temperature):
                current_route = np.copy(candidate_route)
                current_distance = candidate_distance

            # Update the best route if a shorter one is found
            if current_distance < best_distance:
                best_route = np.copy(current_route)
                best_distance = current_distance

            # Reduce the temperature
            temperature *= cooling_rate

        return best_route, best_distance

# Function to plot the TSP route and cities
def plot_results(city_coordinates, best_route, best_distance):
    plt.figure(figsize=(8, 6))
    plt.scatter(city_coordinates[:, 0], city_coordinates[:, 1], label='Cities', c='red')
    plt.plot(city_coordinates[best_route, 0], city_coordinates[best_route, 1], linestyle='-', linewidth=2, label='Best Route')
    plt.title(f'Traveling Salesman Problem - Simulated Annealing\nTotal Distance: {best_distance:.2f}')
    plt.legend()
    plt.show()

# Main function:
# The main function instantiates the TravelingSalesmanProblem class with city 
# coordinates from a CSV file, applies the simulated annealing algorithm to find
# the best TSP route and its corresponding distance, and finally, visualizes the
#  results using Matplotlib. The code's modular structure promotes readability
#  and maintainability, providing a clear illustration of how simulated annea-
# ling can be applied to solve combinatorial optimization problems like the 
# Traveling Salesman Problem.
def main():
    # Create an instance of the TravelingSalesmanProblem class
    tsp = TravelingSalesmanProblem('/Users/igormol/Desktop/city_coordinates.csv')
    # Apply simulated annealing to find the best TSP route and distance
    best_route, best_distance = tsp.simulated_annealing()
    # Plot the results
    plot_results(tsp.city_coordinates, best_route, best_distance)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
