# Particle Swarm Algorithm for Combnatorial Optimisation
#
# Igor Mol <igor.mol@makes.ai>
#
# The Particle Swarm Optimization (PSO) algorithm encapsulates a heuristic 
# approach inspired by the communal dynamics observed in avian and piscine 
# species. Herein, PSO is applied to address the venerable Traveling Salesman 
# Problem (TSP), a quintessential combinatorial optimization quandary. The TSP 
# mandates the discovery of an optimal route that systematically visits an array
# of cities precisely once and ultimately returns to the originating city, all
# while minimizing the aggregate distance traversed. The algorithm orchestrates 
# a collective of particles, each emblematic of a prospective solution (route)
# to the TSP. These particles traverse the solution space, perpetually refining
# their positions predicated on their individual experiences (personal best) and
# the collective intelligence of the swarm (global best). Via iterative 
# refinement, the PSO algorithm progressively converges toward an optimal route
# for the TSP.
#     Embarking upon its trajectory, the PSO algorithm initiates a population of
# particles, where each particle typifies a potential route for the TSP. The 
# position of each particle is delineated by a permutation of cities, accompa-
# nied by the assignment of a random velocity. Subsequently, the algorithm 
# scrutinizes the fitness of each particle's route, considering the cumulative 
# distance traversed. Throughout the optimization odyssey, particles recalibrate 
# their velocities and positions through a judicious combination of inertia, 
# cognitive, and social terms. The inertia term steadfastly upholds the parti-
# cle's present trajectory, while the cognitive and social terms guide the 
# particle toward its personal best and the global best routes, respectively. 
# Through the dynamic adjustment of these parameters over iterative cycles, the 
# swarm collectively hones its exploration of the solution space, converging 
# with metronomic precision towards an optimal TSP route.
#     The presented code encompasses functionalities for the importation of city 
# coordinates, the computation of inter-city distances, and the graphical 
# delineation of the optimized route. Users are afforded the latitude to 
# stipulate the number of cities, particles, and iterations. The algorithm 
# achieves convergence towards an optimal solution by dynamically modulating the
# velocities and positions of its constituent particles. The resultant outcomes,
# encapsulating the optimal route and aggregate distance, are delineated for 
# perspicacity into the efficiency of the optimized TSP solution. Furthermore,
# a visual tableau of the optimal route is instantiated employing matplotlib, 
# endowing stakeholders with an illustrative comprehension of the algorithm's
# operational efficacy. This comprehensive exposition elucidates the nuances 
# inherent in the PSO algorithm and its efficacious application to resolving the
#  Traveling Salesman Problem.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Particle class represents a solution in the problem space
class Particle:
    def __init__(self, num_cities):
        # Initialize a random permutation of cities as a route
        self.route = np.random.permutation(np.arange(num_cities))
        # Initialize a random velocity for each city in the route
        self.velocity = np.random.rand(num_cities)
        # Set the initial personal best as the current route
        self.personal_best_route = np.copy(self.route)
        # Set the initial personal best distance as positive infinity
        self.personal_best_distance = float('inf')

# Function to load city coordinates from a CSV file
def load_coordinates_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df[['X', 'Y']].values

# Function to calculate distances between all pairs of cities
def calculate_distances(coordinates):
    return np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)

# Function to calculate the total distance of a given route
def total_distance(route, distances):
    return np.sum(distances[route[:-1], route[1:]]) + distances[route[-1], route[0]]

# Function to update the velocity of a particle based on PSO formulas
def update_velocity(particle, global_best_route, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
    inertia_term = inertia_weight * particle.velocity
    cognitive_term = cognitive_weight * np.random.rand() * (particle.personal_best_route - particle.route)
    social_term = social_weight * np.random.rand() * (global_best_route - particle.route)
    return inertia_term + cognitive_term + social_term

# PSO algorithm to find the optimal route for the Traveling Salesman Problem
def particle_swarm_optimization(num_cities, num_particles=50, num_iterations=1000):
    # Load city coordinates and calculate distances
    city_coordinates = load_coordinates_from_csv('/Users/igormol/Desktop/city_coordinates.csv')
    distances = calculate_distances(city_coordinates)

    # Initialize particles with random routes
    particles = [Particle(num_cities) for _ in range(num_particles)]

    # Find the initial global best route and distance
    global_best_route = min(particles, key=lambda particle: particle.personal_best_distance).personal_best_route
    global_best_distance = total_distance(global_best_route, distances)

    # PSO main loop
    for _ in range(num_iterations):
        for particle in particles:
            # Update particle velocity and route
            particle.velocity = update_velocity(particle, global_best_route)
            particle.route = np.argsort(particle.velocity)
            current_distance = total_distance(particle.route, distances)

            # Update personal best if the current route is better
            if current_distance < particle.personal_best_distance:
                particle.personal_best_route = np.copy(particle.route)
                particle.personal_best_distance = current_distance

            # Update global best if the current route is better
            if current_distance < global_best_distance:
                global_best_route = np.copy(particle.route)
                global_best_distance = current_distance

    return global_best_route, global_best_distance

# Function to plot the final route on a scatter plot
def plot_route(city_coordinates, route, title="TSP Solution"):
    plt.figure(figsize=(8, 6))
    plt.scatter(city_coordinates[:, 0], city_coordinates[:, 1], label='Cities', c='red')
    plt.plot(city_coordinates[route, 0], city_coordinates[route, 1], linestyle='-', linewidth=2, label='Route')
    plt.title(title)
    plt.legend()
    plt.show()

# Main function
def main():
    np.random.seed(42)

    num_cities = 10  # Adjust the number of cities as needed
    # Run PSO algorithm and get the best route and distance
    best_route, best_distance = particle_swarm_optimization(num_cities)

    # Print the results
    print(f"Best Route: {best_route}")
    print(f"Total Distance: {best_distance:.2f}")

    # Load city coordinates and plot the final route
    city_coordinates = load_coordinates_from_csv('/Users/igormol/Desktop/city_coordinates.csv')
    plot_route(city_coordinates, best_route, title="TSP Solution - Particle Swarm Optimization")

if __name__ == "__main__":
    main()
